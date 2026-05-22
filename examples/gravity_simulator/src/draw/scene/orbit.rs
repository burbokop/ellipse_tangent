use crate::{draw::common::draw_fading_ellipse, orbit::EllipticOrbit};
use burbomath::{NonNeg, Pi, Two};
use core::f32;
use nannou::{color::Rgb, Draw};

pub fn draw_elliptic_orbit(
    draw: &Draw,
    orbit: &EllipticOrbit,
    color: Rgb,
    compensatory_scale: f32,
) {
    draw_fading_ellipse(
        draw,
        &orbit.ellipse,
        orbit.anomaly.radians() / (NonNeg::<f32>::two() * NonNeg::<f32>::pi()),
        color,
        compensatory_scale,
    );
}
