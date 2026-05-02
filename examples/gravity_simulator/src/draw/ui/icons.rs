use burbomath::{Angle, Point, Vector};
use nannou::{
    color::{Rgba, Rgba8},
    Draw,
};

pub fn draw_prograde_icon(
    draw: &Draw,
    center: Point<f32>,
    color: Rgba8,
    radius: f32,
    compensatory_scale: f32,
) {
    draw.x(*center.x())
        .y(*center.y())
        .scale(compensatory_scale)
        .ellipse()
        .stroke(color)
        .stroke_weight(1.)
        .radius(radius)
        .color(Rgba::from_components((0., 0., 0., 0.)));

    draw.x(*center.x())
        .y(*center.y())
        .scale(compensatory_scale)
        .ellipse()
        .radius(radius / 4.)
        .color(color);
}

pub fn draw_retrograde_icon(
    draw: &Draw,
    center: Point<f32>,
    color: Rgba8,
    radius: f32,
    compensatory_scale: f32,
) {
    let radius_sqrt = (4. * radius).sqrt();

    draw.x(*center.x())
        .y(*center.y())
        .scale(compensatory_scale)
        .ellipse()
        .stroke(color)
        .stroke_weight(1.)
        .radius(radius)
        .color(Rgba::from_components((0., 0., 0., 0.)));

    draw.line()
        .points(
            (center.x() + radius_sqrt, center.y() + radius_sqrt).into(),
            (center.x() - radius_sqrt, center.y() - radius_sqrt).into(),
        )
        .color(color)
        .weight(1.);

    draw.line()
        .points(
            (center.x() + radius_sqrt, center.y() - radius_sqrt).into(),
            (center.x() - radius_sqrt, center.y() + radius_sqrt).into(),
        )
        .color(color)
        .weight(1.);
}

pub fn draw_radial_in_icon(
    draw: &Draw,
    center: Point<f32>,
    color: Rgba8,
    radius: f32,
    compensatory_scale: f32,
) {
    let radius_sqrt = (4. * radius).sqrt();
    let inner_radius_sqrt = (radius / 2.).sqrt();

    draw.x(*center.x())
        .y(*center.y())
        .scale(compensatory_scale)
        .ellipse()
        .stroke(color)
        .stroke_weight(1.)
        .radius(radius)
        .color(Rgba::from_components((0., 0., 0., 0.)));

    draw.line()
        .points(
            (center.x() + radius_sqrt, center.y() + radius_sqrt).into(),
            (
                center.x() + inner_radius_sqrt,
                center.y() + inner_radius_sqrt,
            )
                .into(),
        )
        .color(color)
        .weight(1.);

    draw.line()
        .points(
            (
                center.x() - inner_radius_sqrt,
                center.y() - inner_radius_sqrt,
            )
                .into(),
            (center.x() - radius_sqrt, center.y() - radius_sqrt).into(),
        )
        .color(color)
        .weight(1.);

    draw.line()
        .points(
            (center.x() + radius_sqrt, center.y() - radius_sqrt).into(),
            (
                center.x() + inner_radius_sqrt,
                center.y() - inner_radius_sqrt,
            )
                .into(),
        )
        .color(color)
        .weight(1.);

    draw.line()
        .points(
            (
                center.x() - inner_radius_sqrt,
                center.y() + inner_radius_sqrt,
            )
                .into(),
            (center.x() - radius_sqrt, center.y() + radius_sqrt).into(),
        )
        .color(color)
        .weight(1.);
}

pub fn draw_radial_out_icon(
    draw: &Draw,
    center: Point<f32>,
    color: Rgba8,
    radius: f32,
    compensatory_scale: f32,
) {
    let radius_sqrt = (4. * radius).sqrt();
    let outer_radius_sqrt = (8. * radius).sqrt();

    draw.x(*center.x())
        .y(*center.y())
        .scale(compensatory_scale)
        .ellipse()
        .stroke(color)
        .stroke_weight(1.)
        .radius(radius)
        .color(Rgba::from_components((0., 0., 0., 0.)));

    draw.x(*center.x())
        .y(*center.y())
        .scale(compensatory_scale)
        .ellipse()
        .radius(radius / 4.)
        .color(color);

    draw.line()
        .points(
            (center.x() + radius_sqrt, center.y() + radius_sqrt).into(),
            (
                center.x() + outer_radius_sqrt,
                center.y() + outer_radius_sqrt,
            )
                .into(),
        )
        .color(color)
        .weight(1.);

    draw.line()
        .points(
            (
                center.x() - outer_radius_sqrt,
                center.y() - outer_radius_sqrt,
            )
                .into(),
            (center.x() - radius_sqrt, center.y() - radius_sqrt).into(),
        )
        .color(color)
        .weight(1.);

    draw.line()
        .points(
            (center.x() + radius_sqrt, center.y() - radius_sqrt).into(),
            (
                center.x() + outer_radius_sqrt,
                center.y() - outer_radius_sqrt,
            )
                .into(),
        )
        .color(color)
        .weight(1.);

    draw.line()
        .points(
            (
                center.x() - outer_radius_sqrt,
                center.y() + outer_radius_sqrt,
            )
                .into(),
            (center.x() - radius_sqrt, center.y() + radius_sqrt).into(),
        )
        .color(color)
        .weight(1.);
}

pub fn draw_maneuver_icon(
    draw: &Draw,
    center: Point<f32>,
    color: Rgba8,
    radius: f32,
    compensatory_scale: f32,
) {
    // let radius_sqrt_x = (3_f32).sqrt() * 2. * radius;
    // let radius_sqrt_y = 1./2. * radius;
    // let inner_radius_sqrt_x = (3_f32).sqrt() * 2. * radius / 2.;
    // let inner_radius_sqrt_y = 1./2. * radius / 2. ;

    let outer0 = Vector::from_polar(radius * 1.5, Angle::from_degrees(30_f32));
    let outer1 = Vector::from_polar(radius * 1.5, Angle::from_degrees(150_f32));

    let inner0 = Vector::from_polar(radius / 2., Angle::from_degrees(30_f32));
    let inner1 = Vector::from_polar(radius / 2., Angle::from_degrees(150_f32));

    draw.x(*center.x())
        .y(*center.y())
        .scale(compensatory_scale)
        .ellipse()
        .stroke(color)
        .stroke_weight(1.)
        .radius(radius)
        .color(Rgba::from_components((0., 0., 0., 0.)));

    draw.x(*center.x())
        .y(*center.y())
        .scale(compensatory_scale)
        .ellipse()
        .radius(radius / 4.)
        .color(color);

    draw.line()
        .points(
            (*center.x(), center.y() + radius * 1.5).into(),
            (*center.x(), center.y() + radius / 2.).into(),
        )
        .color(color)
        .weight(1.);

    draw.line()
        .points(
            (center.x() - inner0.x(), center.y() - inner0.y()).into(),
            (center.x() - outer0.x(), center.y() - outer0.y()).into(),
        )
        .color(color)
        .weight(1.);

    draw.line()
        .points(
            (center.x() - inner1.x(), center.y() - inner1.y()).into(),
            (center.x() - outer1.x(), center.y() - outer1.y()).into(),
        )
        .color(color)
        .weight(1.);
}

pub fn draw_heading_icon(
    draw: &Draw,
    center: Point<f32>,
    color: Rgba8,
    radius: f32,
    compensatory_scale: f32,
) {
    draw.x(*center.x())
        .y(*center.y())
        .scale(compensatory_scale)
        .ellipse()
        .radius(radius / 4.)
        .color(color);
}
