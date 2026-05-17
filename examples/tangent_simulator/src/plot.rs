use ellipse_tangent::{
    ellipse::TangentDirection,
    line::Line,
    utils::{deg_to_rad_f64, mul_tuple2_f64},
};
use nannou::{
    color::{IntoLinSrgba, BLACK, BLUE, LIGHTBLUE, LIGHTPINK, RED, VIOLET, WHITE, YELLOW},
    draw::properties::ColorScalar,
    event::ElementState,
    geom::pt2,
    math::num_traits::Pow,
    window::Id,
    App, Draw, Frame,
};

use crate::Model;

pub fn new_plot_window(app: &App) -> Id {
    app.new_window()
        .title("window b")
        .size(500, 500)
        .view(view::<rand::rngs::ThreadRng>)
        .raw_event(raw_window_event::<rand::rngs::ThreadRng>)
        .build()
        .unwrap()
}

fn raw_window_event<R: rand::RngCore>(
    app: &App,
    model: &mut Model<R>,
    event: &nannou::winit::event::WindowEvent,
) {
    if let Some(window) = app.window(model.windows.plot_window) {
        let window_rect = window.rect();
        let window_scale_factor = window.scale_factor();

        match event {
            nannou::winit::event::WindowEvent::MouseWheel {
                device_id,
                delta,
                phase,
                modifiers,
            } => {
                let sensitivity = 1.1_f64;
                match delta {
                    nannou::event::MouseScrollDelta::LineDelta(_, y) => {
                        if model.plot_magnification_change_axis_y {
                            model.plot_magnification.0 *= sensitivity.pow(y)
                        } else {
                            model.plot_magnification.1 *= sensitivity.pow(y)
                        }
                    }
                    nannou::event::MouseScrollDelta::PixelDelta(_) => todo!(),
                }
            }
            nannou::winit::event::WindowEvent::MouseInput {
                device_id,
                state,
                button,
                modifiers,
            } => match state {
                ElementState::Pressed => model.plot_magnification_change_axis_y = true,
                ElementState::Released => model.plot_magnification_change_axis_y = false,
            },
            _ => {}
        }
    }
}

fn draw_plot<C>(
    draw: &Draw,
    fun: impl Fn(f64) -> (f64, f64),
    colors: [C; 2],
    magnification: (f64, f64),
    current_k: f64,
    common_tangents: &[(Line<f64>, TangentDirection)],
) where
    C: IntoLinSrgba<ColorScalar> + Clone,
{
    let mut prev_k = 0.;
    let mut prev = (0., 0.);
    let mut has_prev: bool = false;
    for i in -500..500 {
        let k = i as f64 / magnification.0;
        let x = i as f32;
        let v = fun(k);
        let v = (v.0 * magnification.1, v.1 * magnification.1);

        if has_prev {
            {
                let p0 = pt2(prev_k, prev.0 as f32);
                let p1 = pt2(x, v.0 as f32);
                assert!(p0.x.is_finite());
                assert!(p0.y.is_finite());
                assert!(p1.x.is_finite());
                assert!(p1.y.is_finite());
                draw.line().points(p0, p1).color(colors[0].clone());
            }
            {
                let p0 = pt2(prev_k, prev.1 as f32);
                let p1 = pt2(x, v.1 as f32);
                if p0.x.is_finite() && p0.y.is_finite() && p1.x.is_finite() && p1.y.is_finite() {
                    draw.line().points(p0, p1).color(colors[1].clone());
                }
            }
        }

        prev_k = x;
        prev = v;
        has_prev = true;
    }

    for t in common_tangents {
        let v = fun(t.0.k);
        let x = t.0.k * magnification.0;
        let y = (v.0 * magnification.1, v.1 * magnification.1);

        let color = match t.1 {
            TangentDirection::Left => BLACK,
            TangentDirection::Right => WHITE,
        };

        draw.ellipse()
            .radius(3.)
            .stroke(VIOLET)
            .stroke_weight(1.)
            .x(x as f32)
            .y(y.0 as f32)
            .color(color);

        draw.ellipse()
            .radius(3.)
            .stroke(VIOLET)
            .stroke_weight(1.)
            .x(x as f32)
            .y(y.1 as f32)
            .color(color);
    }

    {
        let current_v = fun(current_k);
        let current_x = current_k * magnification.0;
        let current_y = (current_v.0 * magnification.1, current_v.1 * magnification.1);
        draw.ellipse()
            .radius(3.)
            .x(current_x as f32)
            .y(current_y.0 as f32)
            .color(colors[0].clone());

        draw.ellipse()
            .radius(3.)
            .x(current_x as f32)
            .y(current_y.1 as f32)
            .color(colors[1].clone());
    }
}

fn view<R: rand::RngCore>(app: &App, model: &Model<R>, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);
    let rect = app.window(model.windows.plot_window).unwrap().rect();

    let k = deg_to_rad_f64(model.settings.theta).tan();

    draw_plot(
        &draw,
        |k| {
            mul_tuple2_f64(
                model.e0.ellipse.outer_tangents_fun(&model.e1.ellipse, k),
                (2., 2.),
            )
        },
        [RED, LIGHTPINK],
        model.plot_magnification,
        k,
        &model.common_tangents,
    );
    draw_plot(
        &draw,
        |k| model.e0.ellipse.tangent_k_alg(&model.e1.ellipse, k),
        [BLUE, LIGHTBLUE],
        model.plot_magnification,
        k,
        &model.common_tangents,
    );

    draw.line()
        .points(pt2(rect.x.start, 0.), pt2(rect.x.end, 0.))
        .color(YELLOW);

    draw.to_frame(app, &frame).unwrap();
}
