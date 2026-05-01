use std::f32::consts::PI;

use burbomath::{Angle, Complex};
use ellipse_tangent::ellipse::Ellipse;
use nannou::{
    color::{Rgb, Rgba},
    geom::pt2,
    math::map_range,
    Draw,
};

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
