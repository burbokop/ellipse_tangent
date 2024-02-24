use std::time::Duration;

mod ellipse;
use ellipse::Ellipse;

fn deg_to_rad(deg: f32) -> f32 {
    deg * std::f32::consts::PI / 180.
}

use nannou::{prelude::*};

struct Model {
    e0: Ellipse,
    e1: Ellipse,
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    let _w_id = app
        .new_window()
        .title("OSC Receiver")
        .size(1000, 480)
        .view(view)
        .build()
        .unwrap();

    Model {
        e0: Ellipse::new(100., 100., 40., 70., deg_to_rad(15.)),
        e1: Ellipse::new(-30., -100., 20., 80., deg_to_rad(300.)),
    }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {}

fn draw_ellipse(draw: &Draw, ellipse: &Ellipse) {
    draw.ellipse()
        .x(ellipse.x)
        .y(ellipse.y)
        .w(ellipse.a * 2.)
        .h(ellipse.b * 2.)
        .color(WHITE)
        .rotate(-f32::atan2(ellipse.i, ellipse.r));

    for y in (ellipse.y - ellipse.b * 4.) as i32..(ellipse.y + ellipse.b * 4.) as i32 {
        for x in (ellipse.x - ellipse.a * 16.) as i32..(ellipse.x + ellipse.a * 16.) as i32 {
            let x = x as f32;
            let y = y as f32;

            if ellipse.eq()(x, y).abs() < 0.04 {
                draw.ellipse().color(DARKCYAN).radius(1.).x(x).y(y);
            }
        }
    }
}

fn draw_common_intersections(draw: &Draw, ellipse0: &Ellipse, ellipse1: &Ellipse, t: f32) {
    let tt = 50;
    let start = pt2(-400., -400.);
    let end = pt2(400., 400.);

    for k in -tt as i32..tt as i32 {
        for d in (start.y as i32..end.y as i32).step_by(10) {
            let k = 1. / (k as f32);
            let d = d as f32;

            let p0 = pt2(start.x, k * start.x + d);
            let p1 = pt2(end.x, k * end.x + d);

            let mut yes = (false, false);

            if ellipse0.intersection_discriminant(k, d).abs() < 1./t.pow(2.) {
                yes.0 = true;
            }
            if ellipse1.intersection_discriminant(k, d).abs() < 1./t.pow(2.) {
                yes.1 = true;
            }

            if yes.0 && yes.1 {
                draw.line().points(p0, p1).color(PURPLE).stroke_weight(2.);
            }
        }
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    // Begin drawing
    let win = app.window_rect();
    let t = app.time;
    let draw = app.draw();

    // Clear the background to black.
    draw.background().color(BLACK);

    draw_ellipse(&draw, &model.e0);
    draw_ellipse(&draw, &model.e1);

    draw_common_intersections(&draw, &model.e0, &model.e1, t);

    // Write the result of our drawing to the window's frame.
    draw.to_frame(app, &frame).unwrap();
}
