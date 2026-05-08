use std::time::Duration;

use nannou::Draw;

use crate::{
    draw::{
        common::{draw_vector, draw_vector_with_icon},
        ui::icons::{draw_heading_icon, draw_prograde_icon},
    },
    orbit::EllipticOrbit,
    palette,
    vessel::Vessel,
};

pub fn draw_vessel(
    draw: &Draw,
    vessel: &Vessel,
    orbit: &EllipticOrbit,
    gravitational_constant: f32,
    compensatory_scale: f32,
    duration_since_start: Duration,
) {
    let body = orbit.body.upgrade().unwrap();

    // let f0 = orbit.ellipse.f0();
    // let f1 = orbit.ellipse.f1();

    let p = orbit.ellipse.point_on_ellipse(orbit.anomaly);

    // let acc = orbit
    //     .ellipse
    //     .acc(orbit.anomaly, body.mass.clone(), gravitational_constant);

    let vel =
        orbit
            .ellipse
            .tangential_velocity(orbit.anomaly, body.mass.clone(), gravitational_constant);

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

    // draw_vector(draw, "a", p, acc, palette::RADIAL_COLOR, compensatory_scale);
}
