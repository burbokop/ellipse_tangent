use burbomath::{Abs, Angle, Complex, FromUSize, IsPositive, Log2, NonNeg, Positive, Two};
use num_traits::{Pow, real::Real};
use std::{
    iter::Sum,
    ops::{Div, DivAssign},
};

pub fn deg_to_rad_f32(deg: f32) -> f32 {
    deg * std::f32::consts::PI / 180.
}

pub fn deg_to_rad_f64(deg: f64) -> f64 {
    deg * std::f64::consts::PI / 180.
}

pub fn deg_to_rot_f32(deg: f32) -> Complex<f32> {
    Complex::from_polar(1., Angle::from_degrees(deg))
}

pub fn deg_to_rot_f64(deg: f64) -> Complex<f64> {
    Complex::from_polar(1., Angle::from_degrees(deg))
}

pub fn mul_tuple2_f32(lhs: (f32, f32), rhs: (f32, f32)) -> (f32, f32) {
    (lhs.0 * rhs.0, lhs.1 * rhs.1)
}

pub fn mul_tuple2_f64(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
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

pub fn exp_dst(x: f32) -> f32 {
    let a = x.abs();
    if a >= 1. {
        a * x.signum()
    } else {
        1. / a * x.signum()
    }
}

pub fn notmalize_array_around_one<T, const N: usize>(mut v: [T; N]) -> [T; N]
where
    T: Div<Output = T>
        + DivAssign
        + Abs<Output = NonNeg<T>>
        + FromUSize
        + Pow<T, Output = T>
        + IsPositive
        + Two
        + Clone
        + Sum,
    Positive<T>: Log2<Output = T>,
{
    //let c_0 = max / 1.;
    //let c_1 = 1. / min;

    let c = T::two().pow(
        v.iter()
            .map(|x| Positive::try_from(x.clone().abs()).ok().unwrap().log2())
            .sum::<T>()
            / T::from_usize(N),
    );

    // 100000000 = 8
    // 100 = 2

    // (8 + 2) / 2 = 5;

    // 10^5 = 100000;

    // 100000000 / 100000 = 1000
    // 100 / 100000 = 0.001
    //println!("c: {}", c);

    for i in 0..N {
        v[i] /= c.clone();
    }
    v
}

// max / 1 == 1 / min
//

// arr.max(|x| x * c)
