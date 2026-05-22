use burbomath::{Angle, Complex, Ellipse, NonNeg, Point, Vector};
use nannou::{
    color::{IntoLinSrgba, Rgb, Rgba},
    draw::properties::ColorScalar,
    geom::pt2,
    math::map_range,
    Draw,
};
use std::{f32::consts::PI, time::Duration};

/// t - from 0 to 1
pub(crate) fn draw_fading_ellipse(
    draw: &Draw,
    ellipse: &Ellipse<f32>,
    t: NonNeg<f32>,
    color: Rgb,
    compensatory_scale: f32,
) {
    if ellipse.x().is_finite() && ellipse.y().is_finite() {
        assert!(ellipse.a().is_finite());
        assert!(ellipse.b().is_finite());
        assert!(ellipse.r().is_finite());
        assert!(ellipse.i().is_finite());
        assert!(t.into_inner().is_finite());
        assert!(compensatory_scale.is_finite());

        let radius_x = *ellipse.a();
        let radius_y = *ellipse.b();

        let num_points: usize = 1000; // Resolution of the ellipse

        // Generate points and colors
        let points = (0..=num_points).map(|i| {
            // Angle from 0 to 2*PI
            let angle = Angle::from_radians(map_range(i, 0, num_points, 0.0, PI * 2.0));

            // Ellipse formula

            let pos = Complex::from_uneven_polar((radius_x, radius_y).into(), angle)
                * Complex::from_cartesian(*ellipse.r(), *ellipse.i())
                * Complex::from_cartesian(0., 1.)
                + Complex::from_cartesian(*ellipse.x(), *ellipse.y());

            // Color changes with angle (0.0 to 1.0)
            // [See Nannou HSL color documentation](https://docs.rs)
            let point_time = i as f32 / num_points as f32;
            let time = (-t.into_inner() - point_time).rem_euclid(1.);
            let color = Rgba::from_components((color.red, color.green, color.blue, time));

            (pt2(*pos.real(), *pos.imag()), color)
        });

        // Draw the path with vertex-specific colors
        draw.polyline()
            .weight(compensatory_scale)
            .points_colored(points);
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

        let icon_radius = f32::min(vec.len().into_inner() / 2., icon_radius);

        let points: [nannou::glam::Vec2; 2] = [
            <(f32, f32)>::from(position).into(),
            <(f32, f32)>::from(position + vec - vec.norm() * icon_radius).into(),
        ];

        assert!(points[0].x.is_finite());
        assert!(points[0].y.is_finite());
        assert!(points[1].x.is_finite());
        assert!(points[1].y.is_finite());
        assert!(compensatory_scale.is_finite());

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
