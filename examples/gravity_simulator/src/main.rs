use burbomath::{camera::Camera, DeltaAngle};
use std::{marker::PhantomData, sync::LazyLock};

mod draw;
mod font_provider;
mod md_array;
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
    event,
    image::{DynamicImage, RgbaImage},
    prelude::*,
    text::Font,
};
use nannou_egui::{self, egui, Egui};
use rand::rngs::ThreadRng;

use crate::{
    draw::{draw_scene, draw_ui},
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

struct EventContext {
    control: bool,
    shift: bool,
    mouse_position: Point<i32>,
    mouse_position_in_world_space: Point<f32>,
}

impl Default for EventContext {
    fn default() -> Self {
        Self {
            control: Default::default(),
            shift: Default::default(),
            mouse_position: (0, 0).into(),
            mouse_position_in_world_space: (0., 0.).into(),
        }
    }
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
    event_context: EventContext,
    _r: PhantomData<R>,
}

fn main() {
    nannou::app(model).update(update).event(event).run();
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
        event_context: Default::default(),
        _r: PhantomData::<ThreadRng>::default(),
        vessel: Vessel {
            kinematic_body: KinematicBody::new(DeltaAngle::from_radians(0.01), 0.1, 1000.),
        },
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

fn event<R: rand::RngCore>(app: &App, model: &mut Model<R>, event: Event) {
    match event {
        Event::WindowEvent { id, simple } => {
            let w = app.window(id).unwrap();

            match simple {
                Some(e) => match e {
                    Moved(vec2) => todo!(),
                    KeyPressed(event::Key::LShift | event::Key::RShift) => {
                        model.event_context.shift = true
                    }
                    KeyReleased(event::Key::LShift | event::Key::RShift) => {
                        model.event_context.shift = false
                    }
                    KeyPressed(event::Key::LControl | event::Key::RControl) => {
                        model.event_context.control = true
                    }
                    KeyReleased(event::Key::LControl | event::Key::RControl) => {
                        model.event_context.control = false
                    }
                    KeyPressed(..) => {}
                    KeyReleased(..) => {}
                    ReceivedCharacter(_) => todo!(),
                    MouseMoved(vec2) => {
                        model.event_context.mouse_position = (vec2.x as i32, vec2.y as i32).into();
                        model.event_context.mouse_position_in_world_space =
                            &(!&model.camera.transformation()).unwrap()
                                * &model.event_context.mouse_position.as_f32();

                        // model.cursor_pos = pt2(
                        //     mouse_position.x as f32 / window_scale_factor as f32 + window_rect.x.start,
                        //     -mouse_position.y as f32 / window_scale_factor as f32 - window_rect.y.start,
                        // );

                        if !model
                            .e0
                            .update(model.event_context.mouse_position_in_world_space)
                        {
                            model
                                .e1
                                .update(model.event_context.mouse_position_in_world_space);
                        }
                    }
                    MousePressed(button) => {
                        return;

                        if model.e0.ellipse.eq()(
                            *model.event_context.mouse_position_in_world_space.x() as f32,
                            *model.event_context.mouse_position_in_world_space.y() as f32,
                        ) < 0.
                        {
                            match button {
                                MouseButton::Left => model.e0.is_grabbed_to_move = true,
                                MouseButton::Right => model.e0.is_grabbed_to_rotate = true,
                                MouseButton::Middle => model.e0.is_grabbed_to_scale = true,
                                _ => {}
                            }
                        }
                        if model.e1.ellipse.eq()(
                            *model.event_context.mouse_position_in_world_space.x() as f32,
                            *model.event_context.mouse_position_in_world_space.y() as f32,
                        ) < 0.
                        {
                            match button {
                                MouseButton::Left => model.e1.is_grabbed_to_move = true,
                                MouseButton::Right => model.e1.is_grabbed_to_rotate = true,
                                MouseButton::Middle => model.e1.is_grabbed_to_scale = true,
                                _ => {}
                            }
                        }
                    }
                    MouseReleased(mouse_button) => {
                        model.e0.is_grabbed_to_move = false;
                        model.e0.is_grabbed_to_rotate = false;
                        model.e0.is_grabbed_to_scale = false;
                        model.e1.is_grabbed_to_move = false;
                        model.e1.is_grabbed_to_rotate = false;
                        model.e1.is_grabbed_to_scale = false;
                    }
                    MouseEntered => {}
                    MouseExited => {}
                    MouseWheel(mouse_scroll_delta, touch_phase) => {
                        let delta_to_y = |a: MouseScrollDelta| -> f32 {
                            match a {
                                MouseScrollDelta::LineDelta(_, y) => y,
                                MouseScrollDelta::PixelDelta(physical_position) => todo!(),
                            }
                        };

                        let angle_delta_to_scale_division = |angle_delta: f32| {
                            let base: f32 = 1.2;

                            base.powf(angle_delta)
                        };

                        let angle_delta_to_translation_delta = |angle_delta: f32| {
                            let velocity: f32 = 10.; // px per step
                            return velocity * angle_delta;
                        };

                        let position = model.event_context.mouse_position.as_f32();
                        let y = delta_to_y(mouse_scroll_delta);

                        if model.event_context.control {
                            // zoom
                            model.camera.concat_scale_centered(
                                angle_delta_to_scale_division(y),
                                position,
                                position,
                            );
                        } else if model.event_context.shift {
                            // scroll horizontally
                            model
                                .camera
                                .add_translation((angle_delta_to_translation_delta(y), 0.).into());
                        } else {
                            // scroll vertically
                            model
                                .camera
                                .add_translation((0., angle_delta_to_translation_delta(y)).into());
                        }
                    }

                    Resized { .. } => {}
                    HoveredFile(path_buf) => todo!(),
                    DroppedFile(path_buf) => todo!(),
                    HoveredFileCancelled => todo!(),
                    Touch(touch_event) => todo!(),
                    TouchPressure(touchpad_pressure) => todo!(),
                    Focused => {}
                    Unfocused => {}
                    Closed => {}
                },
                None => {}
            }
        }
        Event::DeviceEvent(device_id, device_event) => {}
        Event::Update(update) => {}
        Event::Suspended => println!("Suspended: {:?}", event),
        Event::Resumed => println!("Resumed: {:?}", event),
    }
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
        let dt = update.since_start.as_secs_f32() * *time_speed;

        let e0_focal_len = (model.e0.ellipse.a.pow(2.) - model.e0.ellipse.b.pow(2.)).sqrt();
        let e1_focal_len = (model.e1.ellipse.a.pow(2.) - model.e1.ellipse.b.pow(2.)).sqrt();

        let angular_velocity0 = model.e0.ellipse.angular_velocity(*theta0 / 360., M, G);
        let angular_velocity1 = model.e1.ellipse.angular_velocity(*theta1 / 360., M, G);
        if *theta_auto_change {
            *theta0 += dt * angular_velocity0.degrees();
            *theta1 += dt * angular_velocity1.degrees();

            if settings.thrust_acceleration > 0.
                && (settings.delta_v_len - settings.thrust_acceleration * dt) >= 0.
            {
                let acc = Vector::from_polar(
                    settings.thrust_acceleration,
                    Angle::from_degrees(settings.delta_v_angle),
                );
                println!("acc: {:?}", acc);

                model.e0.ellipse = model.e0.ellipse.accelerated(*theta0 / 360., M, G, dt, acc);
                model.e1.ellipse = model.e1.ellipse.accelerated(*theta1 / 360., M, G, dt, acc);

                settings.delta_v_len -= settings.thrust_acceleration * dt;
                println!("settings.delta_v_len: {}", settings.delta_v_len);
            }
        }

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

fn view<R: rand::RngCore>(app: &App, model: &Model<R>, frame: Frame) {
    let draw = app.draw();

    draw_scene(&draw, model);

    // let win = app.window(model.windows.main_window).unwrap();
    // let window_size = win.inner_size_pixels();
    // let window_rect = (0.,0., window_size.0 as f32, window_size.1 as f32).into();

    let window_rect = nannou_rect_to_rect(app.window_rect());

    draw_ui(&draw, window_rect, model);

    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}
