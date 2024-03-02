use nannou::{
    color::{IntoLinSrgba, BLACK, BLUE, LIGHTBLUE, LIGHTPINK, RED, YELLOW},
    draw::properties::ColorScalar,
    geom::pt2,
    window::Id,
    App, Draw, Frame,
};

use crate::Model;

pub fn new_plot_window(app: &App) -> Id {
    app.new_window()
        .title("window b")
        .size(500, 500)
        .view(view::<rand::rngs::ThreadRng>)
        .build()
        .unwrap()
}

fn draw_plot<C>(draw: &Draw, fun: impl Fn(f32) -> (f32, f32), colors: [C; 2])
where
    C: IntoLinSrgba<ColorScalar> + Clone,
{
    let mut prev_k = 0.;
    let mut prev = (0., 0.);
    let mut has_prev: bool = false;
    for i in -500..500 {
        let k = i as f32 / 10.;
        let x = i as f32;
        let v = fun(k);
        let v = (v.0 / 10., v.1 / 10.);

        if has_prev {
            draw.line()
                .points(pt2(prev_k, prev.0), pt2(x, v.0))
                .color(colors[0].clone());
            draw.line()
                .points(pt2(prev_k, prev.1), pt2(x, v.1))
                .color(colors[1].clone());
        }

        prev_k = x;
        prev = v;
        has_prev = true;
    }
}

fn view<R: rand::RngCore>(app: &App, model: &Model<R>, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);
    let rect = app.window(model.windows.plot_window).unwrap().rect();

    draw_plot(
        &draw,
        |k| model.e0.ellipse.outer_tangents_fun(&model.e1.ellipse, k),
        [RED, LIGHTPINK],
    );
    draw_plot(
        &draw,
        |k| model.e0.ellipse.tangent_k_alg(&model.e1.ellipse, k),
        [BLUE, LIGHTBLUE],
    );

    draw.line()
        .points(pt2(rect.x.start, 0.), pt2(rect.x.end, 0.))
        .color(YELLOW);

    draw.to_frame(app, &frame).unwrap();
}
