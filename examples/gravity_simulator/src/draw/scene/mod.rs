use std::time::Duration;

use crate::{
    draw::{
        common::{draw_fading_ellipse, draw_vector, draw_vector_with_icon},
        ui::icons::{draw_heading_icon, draw_prograde_icon},
    },
    palette,
    utils::{color_from_hex, matrix_to_mat4},
    vessel::Vessel,
    Model, G, PALLETE,
};
use burbomath::{physics::Kg, Angle, Vector};
use ellipse_tangent::ellipse::Ellipse;
use nannou::{
    color::{Alpha, Rgb, BLACK, BLUEVIOLET, CYAN, MAGENTA, RED, YELLOW},
    Draw,
};

fn draw_ellipse(
    draw: &Draw,
    ellipse: &Ellipse,
    vessel: &Vessel,
    t: f32,
    delta_v: Vector<f32>,
    name: &str,
    compensatory_scale: f32,
    celestial_body_mass: Kg<f32>,
    duration_since_start: Duration,
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

    let p = ellipse.point_on_ellipse(Angle::from_degrees(t * 360.));
    draw.x(*p.x())
        .y(*p.y())
        .scale(compensatory_scale)
        .ellipse()
        .radius(focal_point_size)
        .color(CYAN);

    let acc = ellipse.acc(
        Angle::from_degrees(t * 360.),
        celestial_body_mass.clone(),
        G,
    );
    let vel = ellipse.tangential_velocity(
        Angle::from_degrees(t * 360.),
        celestial_body_mass.clone(),
        G,
    );

    draw_vector_with_icon(
        draw,
        draw_heading_icon,
        p,
        vessel.kinematic_body.heading(),
        palette::UI_STROKE_COLOR,
        compensatory_scale,
        duration_since_start,
    );

    draw_vector_with_icon(
        draw,
        draw_prograde_icon,
        p,
        vel,
        palette::PROGRADE_RETROGRADE_COLOR,
        compensatory_scale,
        duration_since_start,
    );

    draw_vector(draw, "a", p, acc, PALLETE[3], compensatory_scale);
    draw_vector(draw, "Δv", p + vel, delta_v, PALLETE[4], compensatory_scale);

    let (_excentricity, new_f1) = ellipse.f1_from_tangential_velocity(
        Angle::from_degrees(t * 360.),
        celestial_body_mass.clone(),
        G,
        vel + delta_v,
    );

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

    let new_p = new_ellipse.point_on_ellipse(Angle::from_degrees(new_t * 360.));
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

pub(crate) fn draw_scene<R: rand::RngCore>(
    draw: &Draw,
    model: &Model<R>,
    duration_since_start: Duration,
) {
    let draw = draw.transform(matrix_to_mat4(model.camera.transformation()));

    draw.background().color(color_from_hex(PALLETE[0]));

    let delta_v = Vector::from_polar(
        model.old_stuff.settings.delta_v_len,
        Angle::from_degrees(model.old_stuff.settings.delta_v_angle),
    );

    let compensatory_scale = 1. / model.camera.transformation().average_scale();

    draw_ellipse(
        &draw,
        &model.vessel_orbit.ellipse,
        &model.vessel,
        model.vessel_orbit.anomaly.degrees() / 360.,
        delta_v,
        "body",
        compensatory_scale,
        model.body.mass.clone(),
        duration_since_start,
    );

    // draw_ellipse(
    //     &draw,
    //     &model.old_stuff.e0.ellipse,
    //     &model.vessel,
    //     model.old_stuff.e0.theta.degrees() / 360.,
    //     delta_v,
    //     "e0",
    //     compensatory_scale,
    //     model.body.mass.clone(),
    // );

    // draw_ellipse(
    //     &draw,
    //     &model.old_stuff.e1.ellipse,
    //     &model.vessel,
    //     model.old_stuff.e1.theta.degrees() / 360.,
    //     delta_v,
    //     "e1",
    //     compensatory_scale,
    //     model.body.mass.clone(),
    // );

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
