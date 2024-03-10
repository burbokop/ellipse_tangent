use ellipse_tangent::{
    ellipse::TangentDirection,
    line::SimpleLine,
    utils::{deg_to_rad, mul_arr, mul_tuple2},
};
use nannou::{
    color::{
        IntoLinSrgba, Srgb, BLACK, BLUE, DARKSLATEGREY, GRAY, GREEN, LIGHTBLUE, LIGHTGRAY,
        LIGHTGREEN, LIGHTPINK, RED, VIOLET, WHITE, YELLOW,
    },
    draw::properties::ColorScalar,
    event::{ElementState, MouseButton},
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
                let sensitivity = 1.1_f32;
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

fn draw_plot<C, const N: usize>(
    draw: &Draw,
    fun: impl Fn(f32) -> [f32; N],
    colors: [C; N],
    magnification: (f32, f32),
    current_k: f32,
    common_tangents: &[(SimpleLine, TangentDirection)],
) where
    C: IntoLinSrgba<ColorScalar> + Clone,
{
    let mut prev_k = 0.;
    let mut prev = [0.; N];
    let mut has_prev: bool = false;
    for i in -500..500 {
        let k = i as f32 / magnification.0;
        let x = i as f32;
        let v = fun(k);

        if has_prev {
            for ((v, prev), color) in v.into_iter().zip(prev).zip(colors.clone()) {
                draw.line()
                    .points(
                        pt2(prev_k, prev * magnification.1),
                        pt2(x, v * magnification.1),
                    )
                    .color(color);
            }
        }

        prev_k = x;
        prev = v;
        has_prev = true;
    }

    for t in common_tangents {
        let v = fun(t.0.k);
        let x = t.0.k * magnification.0;
        let y = (0., 0.);

        let color = match t.1 {
            TangentDirection::Left => BLACK,
            TangentDirection::Right => WHITE,
        };

        draw.ellipse()
            .radius(3.)
            .stroke(VIOLET)
            .stroke_weight(1.)
            .x(x)
            .y(y.0)
            .color(color);

        draw.ellipse()
            .radius(3.)
            .stroke(VIOLET)
            .stroke_weight(1.)
            .x(x)
            .y(y.1)
            .color(color);
    }

    {
        let current_v = fun(current_k);
        let current_x = current_k * magnification.0;
        for (current_v, color) in current_v.into_iter().zip(colors) {
            let current_y = current_v * magnification.1;
            draw.ellipse()
                .radius(3.)
                .x(current_x)
                .y(current_y)
                .color(color);
        }
    }
}

fn view<R: rand::RngCore>(app: &App, model: &Model<R>, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);
    let rect = app.window(model.windows.plot_window).unwrap().rect();

    let k = deg_to_rad(model.settings.theta).tan();

    // draw_plot(
    //     &draw,
    //     |k| {
    //         mul_arr(
    //             model.e0.ellipse.outer_tangents_sdf(&model.e1.ellipse, k),
    //             [2., 2.],
    //         )
    //     },
    //     [RED, LIGHTPINK],
    //     model.plot_magnification,
    //     k,
    //     &model.common_tangents,
    // );

    draw_plot(
        &draw,
        |k| model.e0.ellipse.xx_outer_tangents_sdf(&model.e1.ellipse, k),
        [GREEN, LIGHTGREEN],
        model.plot_magnification,
        k,
        &model.common_tangents,
    );

    // draw_plot(
    //     &draw,
    //     |k| model.e0.ellipse.tangent_k_alg(&model.e1.ellipse, k),
    //     [BLUE, LIGHTBLUE],
    //     model.plot_magnification,
    //     k,
    //     &model.common_tangents,
    // );

    draw.line()
        .points(pt2(rect.x.start, 0.), pt2(rect.x.end, 0.))
        .color(YELLOW);

    draw.to_frame(app, &frame).unwrap();
}
