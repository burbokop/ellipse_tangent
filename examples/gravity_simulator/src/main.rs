#![feature(const_default)]
#![feature(const_trait_impl)]

use burbomath::{camera::Camera, Complex, DeltaAngle, Matrix, NonNeg, Pi};
use std::{
    marker::PhantomData,
    sync::LazyLock,
    time::{Duration, Instant},
};

mod draw;
mod event_handler;
mod font_provider;
mod md_array;
mod palette;
mod plot;
mod utils;
mod vessel;

use burbomath::{Angle, Point, Vector};
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
use nannou_egui::{self, egui, Egui};
use rand::rngs::ThreadRng;

use crate::{
    draw::{
        scene::draw_scene,
        ui::{
            UIData, draw_ui, panels::{
                FlightInfoData, ManueverInfoData, NavCircleData, ThrottleBarData, TimeInfoData, VesselInfoData
            }
        },
    },
    event_handler::EventHandlerContext,
    font_provider::FontProvider,
    utils::nannou_rect_to_rect,
    vessel::{KinematicBody, Vessel},
};

static FP: LazyLock<FontProvider> = LazyLock::new(|| FontProvider::new());
static FONT: LazyLock<Font> = LazyLock::new(|| FP.font());

const M: f32 = 1000_000_000_000.; // kg
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

struct Model<R: rand::RngCore> {
    vessel: Vessel,
    e0: EllipseState,
    e1: EllipseState,
    common_tangents: Vec<(Line, TangentDirection)>,
    settings: Settings,
    egui: Egui,
    image: DynamicImage,
    windows: Windows,
    plot_magnification: (f32, f32),
    plot_magnification_change_axis_y: bool,
    camera: Camera<f32>,
    event_handler_context: EventHandlerContext,
    time_speed: f32,
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

    let ellipse1 = Ellipse::new(-30., -100., 100., 70., deg_to_rad(0.));

    Model {
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
        egui,
        image: DynamicImage::ImageRgba8(RgbaImage::new(
            window.rect().w() as u32,
            window.rect().h() as u32,
        )),
        windows: Windows {
            main_window: main_window_id,
            // plot_window: plot_window_id,
        },
        plot_magnification: (1., 1.),
        plot_magnification_change_axis_y: false,
        camera: Camera::default(),
        event_handler_context: Default::default(),
        _r: PhantomData::<ThreadRng>::default(),
        vessel: Vessel {
            kinematic_body: KinematicBody::new(
                DeltaAngle::from_radians(1.),
                NonNeg::new(0.1).unwrap(),
                NonNeg::new(0.1).unwrap(),
                NonNeg::new(0.05).unwrap(),
                NonNeg::new(1000.).unwrap(),
            ),
        },
        time_speed: 1.,
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
        let settings = &mut model.settings;
        egui.set_elapsed_time(update.since_start);
        let ctx = egui.begin_frame();

        let theta0 = &mut model.e0.theta.degrees();
        let theta1 = &mut model.e1.theta.degrees();
        let theta_auto_change = &mut settings.theta_auto_change;
        let time_speed = &mut settings.time_speed;
        let time_since_start =
            Duration::from_secs_f32(update.since_start.as_secs_f32() * *time_speed);

        let dt = Duration::from_secs_f32(update.since_last.as_secs_f32() * *time_speed);

        println!("dt: {}", dt.as_secs_f32());

        let e0_focal_len = (model.e0.ellipse.a.pow(2.) - model.e0.ellipse.b.pow(2.)).sqrt();
        let e1_focal_len = (model.e1.ellipse.a.pow(2.) - model.e1.ellipse.b.pow(2.)).sqrt();

        let angular_velocity0 = model.e0.ellipse.angular_velocity(*theta0 / 360., M, G);
        let angular_velocity1 = model.e1.ellipse.angular_velocity(*theta1 / 360., M, G);
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

                model.e0.ellipse = model.e0.ellipse.accelerated(
                    *theta0 / 360.,
                    M,
                    G,
                    time_since_start.as_secs_f32(),
                    acc,
                );
                model.e1.ellipse = model.e1.ellipse.accelerated(
                    *theta1 / 360.,
                    M,
                    G,
                    time_since_start.as_secs_f32(),
                    acc,
                );

                settings.delta_v_len -=
                    settings.thrust_acceleration * time_since_start.as_secs_f32();
                println!("settings.delta_v_len: {}", settings.delta_v_len);
            }
        }

        if model.event_handler_context.center_on_vessel_mode() {
            let t = model.e1.theta.degrees() / 360.;
            let target_point = model.e1.ellipse.point_on_ellipse(t);

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

        if model.event_handler_context.x_pressed() {
            model.vessel.kinematic_body.brake_rotation(dt);
        } else if model.event_handler_context.a_pressed() {
            model.vessel.kinematic_body.rotate_left(dt);
        } else if model.event_handler_context.d_pressed() {
            model.vessel.kinematic_body.rotate_right(dt);
        }

        model.vessel.kinematic_body.proceed(dt);

        egui::Window::new("Settings").show(&ctx, |ui| {
            // Scale slider
            ui.label(format!("E0: {:.2?}", &model.e0.ellipse));
            ui.label(format!("E1: {:.2?}", &model.e1.ellipse));

            ui.label(format!("ω0: {:.2?}°", &angular_velocity0.degrees()));
            ui.label(format!("ω1: {:.2?}°", &angular_velocity1.degrees()));
            ui.label("θ0:");
            ui.add(egui::Slider::new(theta0, (0.)..=360.).step_by(1.));
            ui.label("θ1:");
            ui.add(egui::Slider::new(theta1, (0.)..=360.).step_by(1.));

            ui.label("Δv θ:");
            ui.add(egui::Slider::new(&mut settings.delta_v_angle, (0.)..=360.));
            ui.label("|Δv|:");
            ui.add(egui::Slider::new(&mut settings.delta_v_len, 0. ..=1.).step_by(0.01));
            ui.label("ta:");
            ui.add(
                egui::Slider::new(&mut settings.thrust_acceleration, 0. ..=0.01).step_by(0.0001),
            );

            if ui
                .button(if *theta_auto_change { "Stop" } else { "Auto" })
                .clicked()
            {
                *theta_auto_change = !*theta_auto_change;
            }

            if *theta_auto_change {
                ui.label("time speed:");
                ui.add(egui::Slider::new(time_speed, (0.)..=0.001).step_by(0.00001));
            }
        });

        model.e0.theta = Angle::from_degrees(*theta0);
        model.e1.theta = Angle::from_degrees(*theta1);
    }

    model.common_tangents = model.e0.ellipse.common_tangents(&model.e1.ellipse);
}

fn produce_ui_data<R: rand::RngCore>(model: &Model<R>) -> UIData {
    let throttle_bar_data = ThrottleBarData {
        throttle: model.vessel.kinematic_body.thrust().into_inner()
            / model.vessel.kinematic_body.max_thrust().into_inner(),
    };

    let t = model.e1.theta.degrees() / 360.;
    let tangential_velocity = model.e1.ellipse.tangential_velocity(t, M, G);

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
    };

    let manuever_info_data = ManueverInfoData {
        manuever_mode: false,
    };

    let flight_info_data = FlightInfoData {
        velocity: tangential_velocity.len(),
        apoapsis: 10000.,
        periapsis: 2000.,
        delta_v_capacity: 100.,
        delta_v_needed_for_manuever: 100.,
        time_to_next_transition_point: 1000.,
    };

    let vessel_info_data = VesselInfoData {
        thrust: model.vessel.kinematic_body.thrust().into_inner(),
        mass: model.vessel.kinematic_body.mass().into_inner(),
        todo1: 0.,
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
        time_info:time_info_data,
    }
}

fn view<R: rand::RngCore>(app: &App, model: &Model<R>, frame: Frame) {
    let draw = app.draw();

    draw_scene(&draw, model);

    // let win = app.window(model.windows.main_window).unwrap();
    // let window_size = win.inner_size_pixels();
    // let window_rect = (0.,0., window_size.0 as f32, window_size.1 as f32).into();

    let window_rect = nannou_rect_to_rect(app.window_rect());

    let ui_data = produce_ui_data(model);

    draw_ui(&draw, window_rect, &ui_data);

    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}
