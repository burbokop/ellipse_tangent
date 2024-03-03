use std::ops::{Div, DivAssign};

use nannou::math::num_traits::real::Real;

pub fn deg_to_rad(deg: f32) -> f32 {
    deg * std::f32::consts::PI / 180.
}

pub fn deg_to_rot(deg: f32) -> (f32, f32) {
    let rad = deg_to_rad(deg);
    (rad.cos(), rad.sin())
}

pub fn mul_tuple2(lhs: (f32, f32), rhs: (f32, f32)) -> (f32, f32) {
    (lhs.0 * rhs.0, lhs.1 * rhs.1)
}

pub fn notmalize_array<T, const N: usize>(mut v: [T; N]) -> [T; N]
where
    T: PartialOrd,
    T: DivAssign,
    T: Clone,
    T: Real,
{
    let max = *v
        .iter()
        .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
        .unwrap();
    for i in 0..N {
        v[i] /= max.clone();
    }
    v
}
