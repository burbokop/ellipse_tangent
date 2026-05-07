use std::{f32::consts::PI, ops::Deref as _, time::Duration};

use burbomath::{Angle, Complex, Point, Vector};
use ellipse_tangent::{ellipse::Ellipse, line::Line};
use nannou::{
    color::{IntoLinSrgba, Rgb, Rgba},
    draw::{primitive, properties::ColorScalar, Drawing},
    geom::pt2,
    math::map_range,
    Draw,
};

use crate::{utils::color_from_hex, FONT};

pub(crate) fn draw_fading_ellipse(
    draw: &Draw,
    ellipse: &Ellipse,
    t: f32,
    color: Rgb,
    compensatory_scale: f32,
) {
    let radius_x = ellipse.a;
    let radius_y = ellipse.b;

    let num_points: usize = 1000; // Resolution of the ellipse

    // Generate points and colors
    let points = (0..=num_points).map(|i| {
        // Angle from 0 to 2*PI
        let angle = Angle::from_radians(map_range(i, 0, num_points, 0.0, PI * 2.0));

        // Ellipse formula

        let pos = Complex::from_uneven_polar((radius_x, radius_y).into(), angle)
            * Complex::from_cartesian(ellipse.r, ellipse.i)
            * Complex::from_cartesian(0., 1.)
            + Complex::from_cartesian(ellipse.x, ellipse.y);

        // let x = angle.cos() * radius_x + ellipse.x;
        // let y = angle.sin() * radius_y + ellipse.y;

        // Color changes with angle (0.0 to 1.0)
        // [See Nannou HSL color documentation](https://docs.rs)
        let point_time = i as f32 / num_points as f32;

        let time = (-t - point_time).rem_euclid(1.);

        let color = Rgba::from_components((color.red, color.green, color.blue, time));

        // let color = hsla(hue, 1.0, 0.5, 1.0);

        (pt2(*pos.real(), *pos.imag()), color)
    });

    // Draw the path with vertex-specific colors
    draw.polyline()
        .weight(compensatory_scale)
        .points_colored(points);
}

pub fn draw_line_by_kd<'a>(draw: &'a Draw, k: f32, d: f32) -> Drawing<'a, primitive::Line> {
    let start = pt2(-400., -400.);
    let end = pt2(400., 400.);

    let x0 = start.x;
    let x1 = end.x;

    let y0 = k * x0 + d;
    let y1 = k * x1 + d;

    draw.line().points(pt2(x0, y0), pt2(x1, y1))
}

pub fn draw_line<'a>(draw: &'a Draw, line: Line) -> Drawing<'a, primitive::Line> {
    draw_line_by_kd(draw, line.k, line.d)
}

pub fn draw_vector(
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

pub fn draw_vector_with_icon<C>(
    draw: &Draw,
    icon: fn(
        &Draw,
        center: Point<f32>,
        color: C,
        radius: f32,
        compensatory_scale: f32,
        duration_since_start: Duration,
    ),
    position: Point<f32>,
    vec: Vector<f32>,
    color: C,
    compensatory_scale: f32,
    duration_since_start: Duration,
) where
    C: IntoLinSrgba<ColorScalar> + Clone,
{
    if position.x().is_finite()
        && position.y().is_finite()
        && vec.x().is_finite()
        && vec.y().is_finite()
    {
        let icon_radius = 8. * compensatory_scale;

        let icon_radius = f32::min(vec.len() / 2., icon_radius);

        let points = [
            <(f32, f32)>::from(position).into(),
            <(f32, f32)>::from(position + vec - vec.norm() * icon_radius).into(),
        ];

        draw.line()
            .points(points[0], points[1])
            .weight(1. * compensatory_scale)
            .color(color.clone());

        icon(
            draw,
            position + vec,
            color,
            icon_radius / compensatory_scale,
            compensatory_scale,
            duration_since_start,
        );
    }
}
