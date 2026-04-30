use burbomath::camera::Camera;
use std::{marker::PhantomData, ops::Deref, sync::LazyLock};

mod font_provider;
mod md_array;
mod plot;

use burbomath::{Angle, Matrix, Point, Vector};
use ellipse_tangent::{
    ellipse::{Ellipse, TangentDirection},
    line::Line,
    utils::deg_to_rot,
};
use nannou::{
    color::Alpha,
    draw::{primitive, Drawing},
    event,
    image::{DynamicImage, RgbaImage},
    prelude::*,
    text::Font,
};
use nannou_egui::{self, egui, Egui};
use rand::rngs::ThreadRng;

use crate::font_provider::FontProvider;

static FP: LazyLock<FontProvider> = LazyLock::new(|| FontProvider::new());
static FONT: LazyLock<Font> = LazyLock::new(|| FP.font());

pub fn color_from_hex(c: u32) -> Rgba8 {
    let a = (c >> 24) as u8;
    let r = (c >> 16) as u8;
    let g = (c >> 08) as u8;
    let b = (c >> 00) as u8;
    Rgba8::from_components((r, g, b, a))
}

const M: f32 = 1000_000_000_000.; // kg
const G: f32 = 6.67430e-11; // m3 * kg^(−1) * s^(−2);
const PALLETE: [u32; 5] = [0xff3D348B, 0xff7678ED, 0xffF7B801, 0xffF18701, 0xffF35B04];

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
            theta_auto_change: false,
            thrust_acceleration: 0.,
            time_speed: 0.000000001,
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

fn draw_line_by_kd<'a>(draw: &'a Draw, k: f32, d: f32) -> Drawing<'a, primitive::Line> {
    let start = pt2(-400., -400.);
    let end = pt2(400., 400.);

    let x0 = start.x;
    let x1 = end.x;

    let y0 = k * x0 + d;
    let y1 = k * x1 + d;

    draw.line().points(pt2(x0, y0), pt2(x1, y1))
}

fn draw_line<'a>(draw: &'a Draw, line: Line) -> Drawing<'a, primitive::Line> {
    draw_line_by_kd(draw, line.k, line.d)
}

fn draw_vector(
    draw: &Draw,
    name: &str,
    position: Point<f32>,
    vec: Vector<f32>,
    color: u32,
    compensatory_scale: f32,
) {
    if position.x().is_finite()
        && position.y().is_finite()
        && vec.x().is_finite()
        && vec.y().is_finite()
    {
        let points = [
            <(f32, f32)>::from(position).into(),
            <(f32, f32)>::from(position + vec).into(),
        ];
        draw.line()
            .points(points[0], points[1])
            .weight(1. * compensatory_scale)
            .color(color_from_hex(color));

        let c = (points[0] + points[1]) / 2.;

        // a⃗;
        draw.x(c.x)
            .y(c.y)
            .scale(compensatory_scale)
            .text(&format!("{}\u{20D7}: {:.2}", name, vec.len()))
            .color(color_from_hex(color))
            .font(FONT.deref().clone());
    }
}

fn draw_ellipse(
    draw: &Draw,
    ellipse: &Ellipse,
    t: f32,
    delta_v: Vector<f32>,
    name: &str,
    compensatory_scale: f32,
) {
    draw.ellipse()
        .x(ellipse.x)
        .y(ellipse.y)
        .w(ellipse.a * 2.)
        .h(ellipse.b * 2.)
        .rotate(-f32::atan2(ellipse.r, ellipse.i))
        .color(color_from_hex(PALLETE[1]));

    let f0 = ellipse.f0();
    let f1 = ellipse.f1();
    let focal_point_size = ellipse.a.abs().min(ellipse.b.abs()) / 10.;

    draw.ellipse()
        .x(*f0.x())
        .y(*f0.y())
        .radius(focal_point_size)
        .color(YELLOW);

    draw.x(*f1.x())
        .y(*f1.y())
        .scale(compensatory_scale)
        .ellipse()
        .radius(focal_point_size)
        .color(BLUEVIOLET);

    let p = ellipse.point_on_ellipse(t);
    draw.x(*p.x())
        .y(*p.y())
        .scale(compensatory_scale)
        .ellipse()
        .radius(focal_point_size)
        .color(CYAN);

    let acc = ellipse.acc(t, M, G);
    let vel = ellipse.tangential_velocity(t, M, G);

    draw_vector(draw, "v", p, vel, PALLETE[2], compensatory_scale);
    draw_vector(draw, "a", p, acc, PALLETE[3], compensatory_scale);
    draw_vector(draw, "Δv", p + vel, delta_v, PALLETE[4], compensatory_scale);

    let (excentricity, new_f1) = ellipse.f1_from_tangential_velocity(t, M, G, vel + delta_v);

    if new_f1.x().is_finite()
        && new_f1.y().is_finite()
        && new_f1.x().abs() < 1000000.
        && new_f1.y().abs() < 1000000.
    {
        draw.x(*new_f1.x())
            .y(*new_f1.y())
            .scale(compensatory_scale)
            .ellipse()
            .radius(focal_point_size)
            .color(MAGENTA);
    } else {
        eprintln!("new_f1 is nan")
    }

    // if excentricity.x().is_finite()
    //     && excentricity.y().is_finite()
    //     && excentricity.x().abs() > 0.00001
    //     && excentricity.y().abs() > 0.00001
    //     && excentricity.x().abs() < 1000000.
    //     && excentricity.y().abs() < 1000000.
    // {
    //     draw.line()
    //         .points(
    //             <(f32, f32)>::from(f0).into(),
    //             <(f32, f32)>::from(f0 + excentricity).into(),
    //         )
    //         .color(RED);
    // }

    let new_ellipse = Ellipse::from_foci(f0, new_f1, p);

    draw.ellipse()
        .x(new_ellipse.x)
        .y(new_ellipse.y)
        .w(new_ellipse.a * 2.)
        .h(new_ellipse.b * 2.)
        .rotate(-f32::atan2(new_ellipse.r, new_ellipse.i))
        .color(Alpha {
            color: RED,
            alpha: 0.4,
        });

    draw.x(ellipse.x)
        .y(ellipse.y)
        .scale(compensatory_scale)
        .text(name)
        .color(BLACK);

    let new_t = t / ellipse.perimeter() * new_ellipse.perimeter();

    let new_p = new_ellipse.point_on_ellipse(new_t);
    draw.x(*new_p.x())
        .y(*new_p.y())
        .scale(compensatory_scale)
        .ellipse()
        .radius(focal_point_size)
        .color(Alpha {
            color: RED,
            alpha: 0.,
        })
        .stroke_weight(1.)
        .stroke_color(RED);
}

fn matrix_to_mat3(x: Matrix<f32>) -> Mat3 {
    let [a, b, c, d, e, f, g, h, i] = x.into();

    // Mat3::from_cols(
    //     (a,b,c).into(),
    //     (d,e,f).into(),
    //     (g,h,i).into()
    // )

    Mat3::from_cols((a, d, g).into(), (b, e, h).into(), (c, f, i).into())
}

fn matrix_to_mat4(x: Matrix<f32>) -> Mat4 {
    let [a, b, c, d, e, f, g, h, i] = x.into();

    Mat4::from_cols(
        (a, b, 0., c).into(),
        (d, e, 0., f).into(),
        (g, h, i, 0.).into(),
        (0., 0., 0., 1.).into(),
    )
    .transpose()
}

fn view<R: rand::RngCore>(app: &App, model: &Model<R>, frame: Frame) {
    let draw = app
        .draw()
        .transform(matrix_to_mat4(model.camera.transformation()));

    draw.background().color(color_from_hex(PALLETE[0]));

    let delta_v = Vector::from_polar(
        model.settings.delta_v_len,
        Angle::from_degrees(model.settings.delta_v_angle),
    );

    let compensatory_scale = 1. / model.camera.transformation().average_scale();

    draw_ellipse(
        &draw,
        &model.e0.ellipse,
        model.e0.theta.degrees() / 360.,
        delta_v,
        "e0",
        compensatory_scale,
    );
    draw_ellipse(
        &draw,
        &model.e1.ellipse,
        model.e1.theta.degrees() / 360.,
        delta_v,
        "e1",
        compensatory_scale,
    );

    //let texture = wgpu::Texture::from_image(app, &model.image);

    //draw.texture(&texture);

    // let k = deg_to_rad(90.-model.settings.theta).tan();

    // let e0d = model.e0.ellipse.tangent_d(k);
    // let e1d = model.e1.ellipse.tangent_d(k);

    // draw_line_by_kd(&draw, k, e0d.0)
    //     .stroke_weight(1.)
    //     .color(GREEN);
    // draw_line_by_kd(&draw, k, e0d.1)
    //     .stroke_weight(1.)
    //     .color(LIGHTGREEN);
    // draw_line_by_kd(&draw, k, e1d.0)
    //     .stroke_weight(1.)
    //     .color(BLUE);
    // draw_line_by_kd(&draw, k, e1d.1)
    //     .stroke_weight(1.)
    //     .color(LIGHTBLUE);

    draw.x(*model.event_context.mouse_position_in_world_space.x())
        .y(*model.event_context.mouse_position_in_world_space.y())
        .scale(compensatory_scale)
        .ellipse()
        .stroke(VIOLET)
        .stroke_weight(2.)
        .radius(5.)
        .color(BLACK);

    // for t in &model.common_tangents {
    //     draw_line(&draw, t.0).stroke_weight(3.).color(VIOLET);
    //     draw_line(&draw, t.0).stroke_weight(1.).color(match t.1 {
    //         TangentDirection::Left => BLACK,
    //         TangentDirection::Right => WHITE,
    //     });
    // }

    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}
