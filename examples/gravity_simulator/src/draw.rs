use std::{f32::consts::PI, ops::Deref as _};

use burbomath::{Angle, Complex, DeltaAngle, Pi, Point, Rect, Size, Sqrt, Vector};
use ellipse_tangent::{ellipse::Ellipse, line::Line};
use nannou::{
    color::{Alpha, Rgb, Rgba, Rgba8, BLACK, BLUEVIOLET, CYAN, MAGENTA, RED, VIOLET, YELLOW},
    draw::{primitive, Drawing},
    geom::pt2,
    math::map_range,
    Draw,
};

use crate::{
    utils::{color_from_hex, matrix_to_mat4},
    Model, FONT, G, M, PALLETE,
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

fn draw_vector_with_icon(
    draw: &Draw,
    icon: fn(&Draw, center: Point<f32>, color: Rgba8, radius: f32, compensatory_scale: f32),
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
        let icon_radius = 8.;

        let points = [
            <(f32, f32)>::from(position).into(),
            <(f32, f32)>::from(position + vec - vec.norm() * icon_radius).into(),
        ];
        let color = color_from_hex(color);
        draw.line()
            .points(points[0], points[1])
            .weight(1. * compensatory_scale)
            .color(color);

        icon(draw, position + vec, color, icon_radius, compensatory_scale);
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
    // draw.ellipse()
    //     .x(ellipse.x)
    //     .y(ellipse.y)
    //     .w(ellipse.a * 2.)
    //     .h(ellipse.b * 2.)
    //     .rotate(-f32::atan2(ellipse.r, ellipse.i))
    //     .stroke_weight(compensatory_scale)
    //     .stroke_color(color_from_hex(PALLETE[1]))
    //     .color(Rgba8::from_components((0,0,0,0)));

    draw_fading_ellipse(
        draw,
        ellipse,
        t,
        Rgb::from_components((0.2, 0.5, 1.)),
        compensatory_scale,
    );

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

    // draw.ellipse()
    //     .x(new_ellipse.x)
    //     .y(new_ellipse.y)
    //     .w(new_ellipse.a * 2.)
    //     .h(new_ellipse.b * 2.)
    //     .rotate(-f32::atan2(new_ellipse.r, new_ellipse.i))
    //     .color(Alpha {
    //         color: RED,
    //         alpha: 0.4,
    //     });

    draw.x(ellipse.x)
        .y(ellipse.y)
        .scale(compensatory_scale)
        .text(name)
        .color(BLACK);

    let new_t = t / ellipse.perimeter() * new_ellipse.perimeter();

    draw_fading_ellipse(
        draw,
        &new_ellipse,
        new_t,
        Rgb::from_components((1., 0.5, 0.3)),
        compensatory_scale,
    );

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

pub(crate) fn draw_scene<R: rand::RngCore>(draw: &Draw, model: &Model<R>) {
    let draw = draw.transform(matrix_to_mat4(model.camera.transformation()));

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

    // draw.x(*model.event_context.mouse_position_in_world_space.x())
    //     .y(*model.event_context.mouse_position_in_world_space.y())
    //     .scale(compensatory_scale)
    //     .ellipse()
    //     .stroke(VIOLET)
    //     .stroke_weight(2.)
    //     .radius(5.)
    //     .color(BLACK);

    // for t in &model.common_tangents {
    //     draw_line(&draw, t.0).stroke_weight(3.).color(VIOLET);
    //     draw_line(&draw, t.0).stroke_weight(1.).color(match t.1 {
    //         TangentDirection::Left => BLACK,
    //         TangentDirection::Right => WHITE,
    //     });
    // }
}

pub struct NavCircleData {
    pub prograde: Vector<f32>,
    pub retrograde: Vector<f32>,
    pub radial_in: Vector<f32>,
    pub radial_out: Vector<f32>,
    pub maneuver: Vector<f32>,
}

fn draw_prograde_icon(
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

fn draw_retrograde_icon(
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

fn draw_radial_in_icon(
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

fn draw_radial_out_icon(
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

fn draw_maneuver_icon(
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

    let outer0 = Vector::from_polar(radius*1.5, Angle::from_degrees(30_f32));
    let outer1 = Vector::from_polar(radius*1.5, Angle::from_degrees(150_f32));

    let inner0 = Vector::from_polar(radius/2., Angle::from_degrees(30_f32));
    let inner1 = Vector::from_polar(radius/2., Angle::from_degrees(150_f32));

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
            (
                center.x() - inner0.x(),
                center.y() - inner0.y(),
            )
                .into(),
            (center.x() - outer0.x(), center.y() - outer0.y()).into(),
        )
        .color(color)
        .weight(1.);


    draw.line()
        .points(
            (
                center.x() - inner1.x(),
                center.y() - inner1.y(),
            )
                .into(),
            (center.x() - outer1.x(), center.y() - outer1.y()).into(),
        )
        .color(color)
        .weight(1.);
}

fn draw_nav_circle(draw: &Draw, bb: Rect<f32>, data: &NavCircleData) {
    let center = bb.center();

    draw.ellipse()
        .x(*center.x())
        .y(*center.y())
        .w(*bb.w())
        .h(*bb.h())
        .color(Rgba::from_components((0., 0., 0., 0.)))
        .stroke_color(BLACK)
        .stroke_weight(1.);

    let radius = f32::min(*bb.w(), *bb.h()) / 2.;

    draw.line()
        .points(
            (bb.left(), *bb.center().y()).into(),
            (bb.right(), *bb.center().y()).into(),
        )
        .color(BLACK)
        .weight(1.);

    draw.line()
        .points(
            (*bb.center().x(), bb.top()).into(),
            (*bb.center().x(), bb.bottom()).into(),
        )
        .color(BLACK)
        .weight(1.);

    draw_vector_with_icon(
        draw,
        draw_prograde_icon,
        center,
        data.prograde.norm() * radius,
        0xffd7fe00,
        1.,
    );

    draw_vector_with_icon(
        draw,
        draw_retrograde_icon,
        center,
        data.retrograde.norm() * radius,
        0xffd7fe00,
        1.,
    );

    draw_vector_with_icon(
        draw,
        draw_radial_in_icon,
        center,
        data.radial_in.norm() * radius,
        0xff00d6d6,
        1.,
    );

    draw_vector_with_icon(
        draw,
        draw_radial_out_icon,
        center,
        data.radial_out.norm() * radius,
        0xff00d6d6,
        1.,
    );

    draw_vector_with_icon(
        draw,
        draw_maneuver_icon,
        center,
        data.maneuver.norm() * radius,
        0xff0000d6,
        1.,
    );
}

pub(crate) fn draw_ui<R: rand::RngCore>(draw: &Draw, window_rect: Rect<f32>, model: &Model<R>) {
    let nav_circle_size: Size<_> = (125., 125.).into();
    let nav_circle_right_margin = 30.;
    let nav_circle_top_margin = 30.;

    let nav_circle_bb = (
        window_rect.right() - nav_circle_size.w() - nav_circle_right_margin,
        window_rect.top() + nav_circle_top_margin,
        *nav_circle_size.w(),
        *nav_circle_size.h(),
    )
        .into();

    let t = model.e0.theta.degrees() / 360.;
    let tangential_velocity = model.e0.ellipse.tangential_velocity(t, M, G);

    let heading = model.vessel.kinematic_body.complex_heading();

    // complex_heading

    let nav_data = NavCircleData {
        prograde: tangential_velocity * heading,
        retrograde: tangential_velocity * heading * Complex::from_polar(1., Pi::pi()),
        radial_in: tangential_velocity * heading * Complex::from_cartesian(0., 1.),
        radial_out: tangential_velocity * heading * Complex::from_cartesian(0., -1.),
        maneuver: (1., 1.).into(),
    };

    draw_nav_circle(draw, nav_circle_bb, &nav_data);
}
