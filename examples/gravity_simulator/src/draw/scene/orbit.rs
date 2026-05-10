use core::f32;

use nannou::{color::Rgb, Draw};

use crate::{draw::common::draw_fading_ellipse, orbit::EllipticOrbit};

pub fn draw_elliptic_orbit(
    draw: &Draw,
    orbit: &EllipticOrbit,
    color: Rgb,
    compensatory_scale: f32,
) {
    draw_fading_ellipse(
        draw,
        &orbit.ellipse,
        orbit.anomaly.radians() / (2. * f32::consts::PI),
        color,
        compensatory_scale,
    );
}
