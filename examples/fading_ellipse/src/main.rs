use burbomath::Ellipse;
use nannou::prelude::*;

fn main() {
    nannou::app(model).update(update).run();
}

struct Model {}

fn model(app: &App) -> Model {
    app.new_window().size(600, 600).view(view).build().unwrap();
    Model {}
}

fn update(_app: &App, _model: &mut Model, _update: Update) {}

fn draw_fading_ellipse(draw: &Draw, ellipse: &Ellipse<f32>, t: f32, compensatory_scale: f32) {
    let radius_x = ellipse.a();
    let radius_y = ellipse.b();

    let num_points = 1000; // Resolution of the ellipse

    // Generate points and colors
    let points = (0..=num_points).map(|i| {
        // Angle from 0 to 2*PI
        let angle = map_range(i, 0, num_points, 0.0, PI * 2.0);

        // Ellipse formula
        let x = angle.cos() * radius_x + ellipse.x();
        let y = angle.sin() * radius_y + ellipse.y();

        // Color changes with angle (0.0 to 1.0)
        // [See Nannou HSL color documentation](https://docs.rs)
        let point_time = i as f32 / num_points as f32;

        let time = (point_time - t).rem_euclid(1.);

        let color = Rgba::from_components((0.2, 0.5, 1., time));

        // let color = hsla(hue, 1.0, 0.5, 1.0);

        (pt2(x, y), color)
    });

    // Draw the path with vertex-specific colors
    draw.polyline()
        .weight(compensatory_scale)
        .points_colored(points);
}

fn view(app: &App, _model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);

    let t = app.time;
    let _win = app.window_rect();

    // Ellipse parameters
    let radius_x = 200.0;
    let radius_y = 100.0;

    let ellipse = Ellipse::from_raw(0., 0., radius_x, radius_y, 1., 0.);

    draw_fading_ellipse(&draw, &ellipse, t / 10., 1.);

    draw.to_frame(app, &frame).unwrap();
}
