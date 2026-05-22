#![feature(const_default)]
#![feature(const_trait_impl)]

mod draw;
mod event_handler;
mod font_provider;
mod manuever;
mod orbit;
mod palette;
mod utils;
mod vessel;

use crate::{
    draw::{
        scene::draw_scene,
        ui::{
            draw_ui,
            panels::{
                ControlsInfoData, FlightInfoData, ManueverInfoData, NavCircleData, ThrottleBarData,
                TimeInfoData, VesselInfoData,
            },
            UIData,
        },
    },
    event_handler::{AutoRotationTarget, EventHandlerContext},
    font_provider::FontProvider,
    manuever::Manuever,
    orbit::{CelestialBody, EllipticOrbit},
    utils::nannou_rect_to_rect,
    vessel::{KinematicBody, Vessel},
};
use burbomath::{
    camera::Camera,
    non_neg,
    physics::{Kg, M, M3},
    time::RelativeDuration,
    Angle, Complex, DeltaAngle, Ellipse, NonNeg, Pi,
};
use core::f32;
use nannou::{prelude::*, text::Font};
use nannou_egui::{self, Egui};
use rand::rngs::ThreadRng;
use std::{marker::PhantomData, rc::Rc, sync::LazyLock, time::Duration};

static FP: LazyLock<FontProvider> = LazyLock::new(|| FontProvider::new());
static FONT: LazyLock<Font> = LazyLock::new(|| FP.font());

const G: f32 = 6.67430e-11; // m3 * kg^(−1) * s^(−2);
const PALLETE: [u32; 5] = [0xff230D4A, 0xff7678ED, 0xffF7B801, 0xffF18701, 0xffF35B04];

struct Model<R: rand::RngCore> {
    egui: Egui,
    vessel: Vessel,
    body: Rc<CelestialBody>,
    vessel_orbit: EllipticOrbit,
    manuever: Option<Manuever>,
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

    let ellipse = Ellipse::from_angle(
        (-30., -100.).into(),
        (2000., 1900.).into(),
        Angle::from_degrees(0_f32),
    );

    let body = Rc::new(CelestialBody::from_density(
        Kg(5513.) / M3(1.),
        M(1000.),
        M(1010.),
        Rgba8::from_components((153, 102, 51, 0xff)),
        Rgba8::from_components((51, 102, 204, 128)),
    ));

    Model {
        egui,
        vessel: Vessel {
            kinematic_body: KinematicBody::new(
                DeltaAngle::from_radians(1.),
                NonNeg::new(0.1 * 100.).unwrap(),
                NonNeg::new(0.1 * 100.).unwrap(),
                NonNeg::new(0.05 * 100.).unwrap(),
                NonNeg::new(1000.).unwrap(),
            ),
        },
        body: body.clone(),
        vessel_orbit: EllipticOrbit {
            body: Rc::downgrade(&body),
            ellipse,
            anomaly: Angle::from_radians(0.),
        },
        manuever: None,
        camera: Camera::default(),
        event_handler_context: Default::default(),
        time_speed: 1.,
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
        egui.set_elapsed_time(update.since_start);
        let _ctx = egui.begin_frame();

        let dt = Duration::from_secs_f32(update.since_last.as_secs_f32() * model.time_speed);

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
            .unwrap();

        if model.event_handler_context.x_pressed() {
            model.vessel.kinematic_body.brake_rotation(dt);
        } else if model.event_handler_context.a_pressed() {
            if model.time_speed <= 1. {
                model.vessel.kinematic_body.rotate_left(dt);
            }
        } else if model.event_handler_context.d_pressed() {
            if model.time_speed <= 1. {
                model.vessel.kinematic_body.rotate_right(dt);
            }
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
                AutoRotationTarget::Maneuver => {
                    let target_angle = model
                        .manuever
                        .as_ref()
                        .expect("Should not enter AutoRotationTarget::Maneuver if manuever is None")
                        .delta_v(&model.vessel_orbit, G)
                        .angle();
                    model.vessel.kinematic_body.rotate_to(target_angle, dt)
                }
            }
        }

        model.vessel.kinematic_body.proceed(dt);

        if model.vessel.kinematic_body.acceleration().into_inner() > f32::EPSILON {
            let delta_v = model
                .vessel_orbit
                .accelerate(model.vessel.kinematic_body.acceleration_vector(), dt);

            if let Some(manuever) = &mut model.manuever {
                manuever.relative_delta_v -= delta_v;
            }
        }

        model.vessel_orbit.proceed(dt);

        if model.event_handler_context.manuever_planner_mode() {
            if let Some(manuever) = &mut model.manuever {
                if model.event_handler_context.less_pressed() {
                    manuever.move_start_anomaly_backward(dt);
                } else if model.event_handler_context.greater_pressed() {
                    manuever.move_start_anomaly_forward(dt);
                } else if model.event_handler_context.left_arrow_pressed() {
                    manuever.accelerate_towards_radial_out(&model.vessel_orbit, dt, G);
                } else if model.event_handler_context.right_arrow_pressed() {
                    manuever.accelerate_towards_radial_in(&model.vessel_orbit, dt, G);
                } else if model.event_handler_context.up_arrow_pressed() {
                    manuever.accelerate_towards_prograde(&model.vessel_orbit, dt, G);
                } else if model.event_handler_context.down_arrow_pressed() {
                    manuever.accelerate_towards_retrograde(&model.vessel_orbit, dt, G);
                }

                manuever.proceed(&model.vessel_orbit, G);
            }
        }
    }
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
        .unwrap();

    let heading = model.vessel.kinematic_body.complex_heading();

    let top_axis = !heading * Complex::from_cartesian(0., 1.);
    let bottom_axis = top_axis * Complex::from_polar(1., Pi::pi());
    let left_axis = top_axis * Complex::from_cartesian(0., -1.);
    let right_axis = top_axis * Complex::from_cartesian(0., 1.);

    let nav_data = NavCircleData {
        prograde: tangential_velocity.norm() * top_axis,
        retrograde: tangential_velocity.norm() * bottom_axis,
        radial_in: tangential_velocity.norm() * left_axis,
        radial_out: tangential_velocity.norm() * right_axis,
        maneuver: if model.event_handler_context.manuever_planner_mode() {
            let manuever = model.manuever.as_ref().unwrap();
            let delta_v = manuever.delta_v(&model.vessel_orbit, G);

            Some(delta_v * top_axis)
        } else {
            None
        },
        auto_rotation_mode: model.event_handler_context.auto_rotation_mode(),
    };

    let manuever_info_data = ManueverInfoData {
        manuever_mode: model.event_handler_context.manuever_planner_mode(),
        time_to_manuever: model
            .manuever
            .as_ref()
            .map(|m| model.vessel_orbit.time_to(m.delta_v_anomaly))
            .unwrap_or(RelativeDuration::from_secs(0)),
        manuever_duration: model
            .manuever
            .as_ref()
            .map(|m| m.duration(model.vessel.kinematic_body.max_acceleration()))
            .unwrap_or(Duration::from_secs(0)),
        delta_v_needed: model
            .manuever
            .as_ref()
            .map(|m| m.relative_delta_v.len())
            .unwrap_or(non_neg!(0.)),
        todo0: 0.,
        todo1: 0.,
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

    let controls_info_data = ControlsInfoData {
        manuever_mode: model.event_handler_context.manuever_planner_mode(),
    };

    UIData {
        nav_circle: nav_data,
        flight_info: flight_info_data,
        manuever_info: manuever_info_data,
        throttle_bar: throttle_bar_data,
        vessel_info: vessel_info_data,
        time_info: time_info_data,
        controls_info: controls_info_data,
    }
}

fn view<R: rand::RngCore>(app: &App, model: &Model<R>, frame: Frame) {
    let draw = app.draw();

    draw_scene(&draw, model, app.duration.since_start);

    let window_rect = nannou_rect_to_rect(app.window_rect());
    let ui_data = produce_ui_data(model);

    draw_ui(&draw, window_rect, &ui_data, app.duration.since_start);

    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}
