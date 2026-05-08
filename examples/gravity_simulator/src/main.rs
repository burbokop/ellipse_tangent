#![feature(const_default)]
#![feature(const_trait_impl)]

mod draw;
mod event_handler;
mod font_provider;
mod md_array;
mod orbit;
mod palette;
mod plot;
mod utils;
mod vessel;

use crate::{
    draw::{
        scene::draw_scene,
        ui::{
            draw_ui,
            panels::{
                FlightInfoData, ManueverInfoData, NavCircleData, ThrottleBarData, TimeInfoData,
                VesselInfoData,
            },
            UIData,
        },
    },
    event_handler::{AutoRotationTarget, EventHandlerContext},
    font_provider::FontProvider,
    orbit::{CelestialBody, EllipticOrbit},
    utils::nannou_rect_to_rect,
    vessel::{KinematicBody, Vessel},
};
use burbomath::{
    camera::Camera,
    physics::{Kg, KgPerM3, M, M3},
    Angle, Complex, DeltaAngle, NonNeg, Pi, Point, Vector,
};
use core::f32;
use ellipse_tangent::{
    ellipse::{Ellipse, TangentDirection},
    line::Line,
    utils::deg_to_rot,
};
use nannou::{
    image::{DynamicImage, RgbaImage},
    prelude::*,
    text::Font,
};
use nannou_egui::{self, Egui};
use rand::rngs::ThreadRng;
use std::{marker::PhantomData, rc::Rc, sync::LazyLock, time::Duration};

static FP: LazyLock<FontProvider> = LazyLock::new(|| FontProvider::new());
static FONT: LazyLock<Font> = LazyLock::new(|| FP.font());

const G: f32 = 6.67430e-11; // m3 * kg^(−1) * s^(−2);
const PALLETE: [u32; 5] = [0xff230D4A, 0xff7678ED, 0xffF7B801, 0xffF18701, 0xffF35B04];

struct Settings {
    delta_v_angle: f32,
    delta_v_len: f32,
    theta_auto_change: bool,
    thrust_acceleration: f32,
    time_speed: f32,
}

struct EllipseState {
    ellipse: Ellipse,
    theta: Angle<f32>,
    is_grabbed_to_move: bool,
    is_grabbed_to_rotate: bool,
    is_grabbed_to_scale: bool,
}

impl EllipseState {
    fn update(&mut self, cursor_pos: Point<f32>) -> bool {
        if self.is_grabbed_to_move {
            self.ellipse.x = *cursor_pos.x();
            self.ellipse.y = *cursor_pos.y();
            true
        } else if self.is_grabbed_to_rotate {
            let rt = deg_to_rot(*cursor_pos.x());
            self.ellipse.r = rt.0;
            self.ellipse.i = rt.1;
            true
        } else if self.is_grabbed_to_scale {
            self.ellipse.a = *cursor_pos.x() / 10.;
            self.ellipse.b = *cursor_pos.y() / 10.;
            true
        } else {
            false
        }
    }
}

struct Windows {
    main_window: WindowId,
    // plot_window: WindowId,
}

struct OldStuff {
    e0: EllipseState,
    e1: EllipseState,
    common_tangents: Vec<(Line, TangentDirection)>,
    settings: Settings,
    image: DynamicImage,
    plot_magnification: (f32, f32),
    plot_magnification_change_axis_y: bool,
}

struct Model<R: rand::RngCore> {
    egui: Egui,
    windows: Windows,
    vessel: Vessel,
    body: Rc<CelestialBody>,
    vessel_orbit: EllipticOrbit,
    camera: Camera<f32>,
    event_handler_context: EventHandlerContext,
    time_speed: f32,

    old_stuff: OldStuff,
    _r: PhantomData<R>,
}

fn main() {
    nannou::app(model)
        .update(update)
        .event(event_handler::event)
        .run();
}

fn model(app: &App) -> Model<impl rand::RngCore> {
    let main_window_id = app
        .new_window()
        .title("Elliptic orbit simulation")
        .size(1400, 700)
        .view(view::<rand::rngs::ThreadRng>)
        .raw_event(raw_window_event::<rand::rngs::ThreadRng>)
        .build()
        .unwrap();

    // let plot_window_id = plot::new_plot_window(app);

    let window = app.window(main_window_id).unwrap();

    let egui = Egui::from_window(&window);

    let ellipse0 = Ellipse::new(100., 100., 40., 70., deg_to_rad(15.));
    // let ellipse1 = Ellipse::new(-30., -100., 20., 80., deg_to_rad(300.));

    let ellipse1 = Ellipse::new(-30., -100., 200., 140., deg_to_rad(0.));

    let body = Rc::new(CelestialBody::from_density(
        Kg(5513.) / M3(1.),
        M(35.),
        M(36.),
        Rgba8::from_components((153, 102, 51, 0xff)),
        Rgba8::from_components((51, 102, 204, 128)),
    ));

    Model {
        egui,
        windows: Windows {
            main_window: main_window_id,
            // plot_window: plot_window_id,
        },
        vessel: Vessel {
            kinematic_body: KinematicBody::new(
                DeltaAngle::from_radians(1.),
                NonNeg::new(0.1 * 10000.).unwrap(),
                NonNeg::new(0.1 * 10000.).unwrap(),
                NonNeg::new(0.05 * 10000.).unwrap(),
                NonNeg::new(1000.).unwrap(),
            ),
        },
        body: body.clone(),
        vessel_orbit: EllipticOrbit {
            body: Rc::downgrade(&body),
            ellipse: ellipse1.clone(),
            anomaly: Angle::from_radians(0.),
        },
        camera: Camera::default(),
        event_handler_context: Default::default(),
        time_speed: 1.,
        old_stuff: OldStuff {
            e0: EllipseState {
                ellipse: ellipse0,
                theta: Angle::from_radians(0.),
                is_grabbed_to_move: false,
                is_grabbed_to_rotate: false,
                is_grabbed_to_scale: false,
            },
            e1: EllipseState {
                ellipse: ellipse1,
                theta: Angle::from_radians(0.),
                is_grabbed_to_move: false,
                is_grabbed_to_rotate: false,
                is_grabbed_to_scale: false,
            },
            common_tangents: vec![],
            settings: Settings {
                delta_v_angle: 0.,
                delta_v_len: 1.,
                theta_auto_change: true,
                thrust_acceleration: 0.,
                time_speed: 0.001,
            },
            image: DynamicImage::ImageRgba8(RgbaImage::new(
                window.rect().w() as u32,
                window.rect().h() as u32,
            )),

            plot_magnification: (1., 1.),
            plot_magnification_change_axis_y: false,
        },
        _r: PhantomData::<ThreadRng>::default(),
    }
}

fn raw_window_event<R: rand::RngCore>(
    _app: &App,
    model: &mut Model<R>,
    event: &nannou::winit::event::WindowEvent,
) {
    // Let egui handle things like keyboard and mouse input.
    model.egui.handle_raw_event(event);
}

fn update<R: rand::RngCore>(app: &App, model: &mut Model<R>, update: Update) {
    {
        let egui = &mut model.egui;
        let settings = &mut model.old_stuff.settings;
        egui.set_elapsed_time(update.since_start);
        let ctx = egui.begin_frame();

        let theta0 = &mut model.old_stuff.e0.theta.degrees();
        let theta1 = &mut model.old_stuff.e1.theta.degrees();

        let theta_auto_change = &mut settings.theta_auto_change;

        let time_speed = &mut settings.time_speed;
        let time_since_start =
            Duration::from_secs_f32(update.since_start.as_secs_f32() * *time_speed);

        let dt = Duration::from_secs_f32(update.since_last.as_secs_f32() * model.time_speed);

        let e0_focal_len =
            (model.old_stuff.e0.ellipse.a.pow(2.) - model.old_stuff.e0.ellipse.b.pow(2.)).sqrt();
        let e1_focal_len =
            (model.old_stuff.e1.ellipse.a.pow(2.) - model.old_stuff.e1.ellipse.b.pow(2.)).sqrt();

        let angular_velocity0 = model.old_stuff.e0.ellipse.angular_velocity(
            Angle::from_degrees(*theta0),
            model.body.mass.clone(),
            G,
        );
        let angular_velocity1 = model.old_stuff.e1.ellipse.angular_velocity(
            Angle::from_degrees(*theta1),
            model.body.mass.clone(),
            G,
        );
        if *theta_auto_change {
            *theta0 += time_since_start.as_secs_f32() * angular_velocity0.degrees();
            *theta1 += time_since_start.as_secs_f32() * angular_velocity1.degrees();

            if settings.thrust_acceleration > 0.
                && (settings.delta_v_len
                    - settings.thrust_acceleration * time_since_start.as_secs_f32())
                    >= 0.
            {
                let acc = Vector::from_polar(
                    settings.thrust_acceleration,
                    Angle::from_degrees(settings.delta_v_angle),
                );
                println!("acc: {:?}", acc);

                model.old_stuff.e0.ellipse = model.old_stuff.e0.ellipse.accelerated(
                    Angle::from_degrees(*theta0),
                    model.body.mass.clone(),
                    G,
                    time_since_start,
                    acc,
                );
                model.old_stuff.e1.ellipse = model.old_stuff.e1.ellipse.accelerated(
                    Angle::from_degrees(*theta1),
                    model.body.mass.clone(),
                    G,
                    time_since_start,
                    acc,
                );

                settings.delta_v_len -=
                    settings.thrust_acceleration * time_since_start.as_secs_f32();
                println!("settings.delta_v_len: {}", settings.delta_v_len);
            }
        }

        if model.event_handler_context.center_on_vessel_mode() {
            let target_point = model
                .vessel_orbit
                .ellipse
                .point_on_ellipse(model.vessel_orbit.anomaly);
            let window_rect = nannou_rect_to_rect(app.window_rect());
            let window_center = window_rect.center();

            model
                .camera
                .translate_to_target(target_point, window_center);
        }

        if model.event_handler_context.w_pressed() {
            model.vessel.kinematic_body.thrust_up(dt);
        } else if model.event_handler_context.s_pressed() {
            model.vessel.kinematic_body.thrust_down(dt);
        } else {
            model.vessel.kinematic_body.brake_thrust_change(dt);
        }

        let tangential_velocity = model
            .vessel_orbit
            .ellipse
            .tangential_velocity(
                model.vessel_orbit.anomaly,
                model.vessel_orbit.body.upgrade().unwrap().mass.clone(),
                G,
            )
            .norm();

        if model.event_handler_context.x_pressed() {
            model.vessel.kinematic_body.brake_rotation(dt);
        } else if model.event_handler_context.a_pressed() {
            model.vessel.kinematic_body.rotate_left(dt);
        } else if model.event_handler_context.d_pressed() {
            model.vessel.kinematic_body.rotate_right(dt);
        } else if let Some(auto_rotation_mode) = model.event_handler_context.auto_rotation_mode() {
            match auto_rotation_mode {
                AutoRotationTarget::Prograde => model
                    .vessel
                    .kinematic_body
                    .rotate_to(tangential_velocity.angle(), dt),
                AutoRotationTarget::Retrograde => model
                    .vessel
                    .kinematic_body
                    .rotate_to(tangential_velocity.angle() + DeltaAngle::<f32>::pi(), dt),
                AutoRotationTarget::RadialIn => model.vessel.kinematic_body.rotate_to(
                    tangential_velocity.angle() - DeltaAngle::<f32>::pi() / 2.,
                    dt,
                ),
                AutoRotationTarget::RadialOut => model.vessel.kinematic_body.rotate_to(
                    tangential_velocity.angle() + DeltaAngle::<f32>::pi() / 2.,
                    dt,
                ),
                AutoRotationTarget::Maneuver => model
                    .vessel
                    .kinematic_body
                    .rotate_to(Angle::from_radians(f32::consts::PI / 4.), dt),
            }
        }

        model.vessel.kinematic_body.proceed(dt);

        if model.vessel.kinematic_body.acceleration().into_inner() > f32::EPSILON {
            model
                .vessel_orbit
                .accelerate(model.vessel.kinematic_body.acceleration_vector(), dt);
        }

        model.vessel_orbit.proceed(dt);

        // egui::Window::new("Settings").show(&ctx, |ui| {
        //     // Scale slider
        //     ui.label(format!("E0: {:.2?}", &model.old_stuff.e0.ellipse));
        //     ui.label(format!("E1: {:.2?}", &model.old_stuff.e1.ellipse));

        //     ui.label(format!("ω0: {:.2?}°", &angular_velocity0.degrees()));
        //     ui.label(format!("ω1: {:.2?}°", &angular_velocity1.degrees()));
        //     ui.label("θ0:");
        //     ui.add(egui::Slider::new(theta0, (0.)..=360.).step_by(1.));
        //     ui.label("θ1:");
        //     ui.add(egui::Slider::new(theta1, (0.)..=360.).step_by(1.));

        //     ui.label("Δv θ:");
        //     ui.add(egui::Slider::new(&mut settings.delta_v_angle, (0.)..=360.));
        //     ui.label("|Δv|:");
        //     ui.add(egui::Slider::new(&mut settings.delta_v_len, 0. ..=1.).step_by(0.01));
        //     ui.label("ta:");
        //     ui.add(
        //         egui::Slider::new(&mut settings.thrust_acceleration, 0. ..=0.01).step_by(0.0001),
        //     );

        //     if ui
        //         .button(if *theta_auto_change { "Stop" } else { "Auto" })
        //         .clicked()
        //     {
        //         *theta_auto_change = !*theta_auto_change;
        //     }

        //     if *theta_auto_change {
        //         ui.label("time speed:");
        //         ui.add(egui::Slider::new(time_speed, (0.)..=0.001).step_by(0.00001));
        //     }
        // });

        model.old_stuff.e0.theta = Angle::from_degrees(*theta0);
        model.old_stuff.e1.theta = Angle::from_degrees(*theta1);
    }

    model.old_stuff.common_tangents = model
        .old_stuff
        .e0
        .ellipse
        .common_tangents(&model.old_stuff.e1.ellipse);
}

fn produce_ui_data<R: rand::RngCore>(model: &Model<R>) -> UIData {
    let throttle_bar_data = ThrottleBarData {
        throttle: model.vessel.kinematic_body.thrust().into_inner()
            / model.vessel.kinematic_body.max_thrust().into_inner(),
    };

    let tangential_velocity = model
        .vessel_orbit
        .ellipse
        .tangential_velocity(model.vessel_orbit.anomaly, model.body.mass.clone(), G)
        .norm();

    let heading = model.vessel.kinematic_body.complex_heading();

    let top_axis = !heading * Complex::from_cartesian(0., 1.);
    let bottom_axis = top_axis * Complex::from_polar(1., Pi::pi());
    let left_axis = top_axis * Complex::from_cartesian(0., -1.);
    let right_axis = top_axis * Complex::from_cartesian(0., 1.);

    let nav_data = NavCircleData {
        prograde: tangential_velocity * top_axis,
        retrograde: tangential_velocity * bottom_axis,
        radial_in: tangential_velocity * left_axis,
        radial_out: tangential_velocity * right_axis,
        maneuver: (1., 1.).into(),
        auto_rotation_mode: model.event_handler_context.auto_rotation_mode(),
    };

    let manuever_info_data = ManueverInfoData {
        manuever_mode: false,
    };

    let flight_info_data = FlightInfoData {
        velocity: tangential_velocity.len(),
        apoapsis: model.vessel_orbit.ellipse.apoapsis(),
        periapsis: model.vessel_orbit.ellipse.periapsis(),
        delta_v_capacity: 100.,
        delta_v_needed_for_manuever: 100.,
        time_to_next_transition_point: 1000.,
    };

    let vessel_info_data = VesselInfoData {
        thrust: model.vessel.kinematic_body.thrust().into_inner(),
        thrust_acceleration: model.vessel.kinematic_body.acceleration().into_inner(),
        mass: model.vessel.kinematic_body.mass().into_inner(),
        todo2: 0.,
        todo3: 0.,
        todo4: 0.,
    };

    let time_info_data = TimeInfoData {
        time_speed: model.time_speed,
    };

    UIData {
        nav_circle: nav_data,
        flight_info: flight_info_data,
        manuever_info: manuever_info_data,
        throttle_bar: throttle_bar_data,
        vessel_info: vessel_info_data,
        time_info: time_info_data,
    }
}

fn view<R: rand::RngCore>(app: &App, model: &Model<R>, frame: Frame) {
    let draw = app.draw();

    draw_scene(&draw, model, app.duration.since_start);

    // let win = app.window(model.windows.main_window).unwrap();
    // let window_size = win.inner_size_pixels();
    // let window_rect = (0.,0., window_size.0 as f32, window_size.1 as f32).into();

    let window_rect = nannou_rect_to_rect(app.window_rect());

    let ui_data = produce_ui_data(model);

    draw_ui(&draw, window_rect, &ui_data, app.duration.since_start);

    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}
