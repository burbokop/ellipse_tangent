use std::ops::DivAssign;
use num_traits::{real::Real, Pow as _};

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

pub fn exp_dst(x: f32) -> f32 {
    let a = x.abs();
    if a >= 1. {
        a * x.signum()
    } else {
        1. / a * x.signum()
    }
}

pub fn notmalize_array_around_one<const N: usize>(mut v: [f32; N]) -> [f32; N] {
    //let c_0 = max / 1.;
    //let c_1 = 1. / min;

    let c = (2_f32).pow(v.iter().map(|x| x.abs().log2()).sum::<f32>() / N as f32);

    // 100000000 = 8
    // 100 = 2

    // (8 + 2) / 2 = 5;

    // 10^5 = 100000;

    // 100000000 / 100000 = 1000
    // 100 / 100000 = 0.001
    //println!("c: {}", c);

    for i in 0..N {
        v[i] /= c;
    }
    v
}

// max / 1 == 1 / min
//

// arr.max(|x| x * c)

pub trait SignedSqr {
    fn ssqr(self) -> Self;
}

impl SignedSqr for f32 {
    fn ssqr(self) -> Self {
        if self >= 0. {
            self.pow(2.)
        } else {
            -self.pow(2.)
        }
    }
}

pub trait SignedSqrt {
    fn ssqrt(self) -> Self;
}

impl SignedSqrt for f32 {
    fn ssqrt(self) -> Self {
        if self >= 0. {
            self.sqrt()
        } else {
            -(-self).sqrt()
        }
    }
}
