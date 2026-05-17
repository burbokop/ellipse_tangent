use burbomath::{
    Abs, Angle, Atan, Complex, Cos, Cube, DeltaAngle, FromDurationAsSecs, FromSecs, FromUSize,
    IsNan, IsNeg, IsPositive, Log2, NonNeg, One, Pi, Point, Positive, RemEuclid, SignedSq,
    SignedSqrt, Sin, Sq, Sqrt, Tan, TopLimit, Two, UnsignedContstant, Vector, Zero, lerp, non_neg,
    physics::Kg, time::RelativeDuration, uconst,
};
use core::f64;
use num_traits::Pow;
use rustnomial::Polynomial;
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, Neg, Sub},
    time::Duration,
};

use crate::{line::Line, utils::notmalize_array_around_one};

#[inline(always)]
fn absmax<T: Abs<Output = NonNeg<T>> + PartialOrd + IsNan + Clone>(a: T, b: T) -> T {
    if a.clone().is_nan() {
        return a;
    }

    if b.clone().is_nan() {
        return b;
    }

    if a.clone().abs() > b.clone().abs() {
        a
    } else {
        b
    }
}

#[inline(always)]
fn absmin<T: Abs<Output = NonNeg<T>> + PartialOrd + IsNan + Clone>(a: T, b: T) -> T {
    if a.clone().is_nan() {
        return a;
    }

    if b.clone().is_nan() {
        return b;
    }

    if a.clone().abs() < b.clone().abs() {
        a
    } else {
        b
    }
}

// pub mod vmath {
//     use num_traits::Pow as _;

//     #[inline(always)]
//     pub fn complex_mul(ab: (T, T), cd: (T, T)) -> (T, T) {
//         let (a, b) = ab;
//         let (c, d) = cd;
//         (a * c - b * d, a * d + b * c)
//     }

//     pub fn add(vec0: (T, T), vec1: (T, T)) -> (T, T) {
//         let (x0, y0) = vec0;
//         let (x1, y1) = vec1;
//         (x0 + x1, y0 + y1)
//     }

//     pub fn sub(vec0: (T, T), vec1: (T, T)) -> (T, T) {
//         let (x0, y0) = vec0;
//         let (x1, y1) = vec1;
//         (x0 - x1, y0 - y1)
//     }

//     pub fn left_perp(vec: (T, T)) -> (T, T) {
//         let (x, y) = vec;
//         (-y, x)
//     }

//     pub fn right_perp(vec: (T, T)) -> (T, T) {
//         let (x, y) = vec;
//         (y, -x)
//     }

//     pub fn len(vec: (T, T)) -> T {
//         let (x, y) = vec;
//         (x.sq() + y.sq()).sqrt()
//     }

//     pub fn norm(vec: (T, T)) -> (T, T) {
//         div(vec, len(vec))
//     }

//     pub fn mul(vec: (T, T), s: T) -> (T, T) {
//         let (x, y) = vec;
//         (x * s, y * s)
//     }

//     pub fn div(vec: (T, T), s: T) -> (T, T) {
//         let (x, y) = vec;
//         (x / s, y / s)
//     }
// }

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse<T> {
    x: T,
    y: T,
    a: T,
    b: T,
    r: T,
    i: T,
}

// (x-x0)^2 / a^2 + (y-y0)^2 / b^2 = 1

//x+yI = (r+iI)*(ux+uyI) = r*ux + r*uy*I + i*I*ux - i*uy = (r*ux - i*uy) + (r*uy + i*ux)*I
//x+yI = (ux+uyI)*(r+iI) = ux*r + i*ux*I + r*I*uy - i*uy = (r*ux - i*uy) + (i*uy + r*ux)*I

// (x-x0)^2 / a^2 + (y-y0)^2 / b^2 = 1

// (r*x - i*y - x0)^2 / a^2 + (r*y + i*x - y0)^2 / b^2 = 1

// (r*(x - x0) - i*(y - y0))^2 / a^2 + (r*(y - y0) + i*(x - x0))^2 / b^2 = 1

#[derive(Debug, Clone)]
pub enum TangentDirection {
    Left,
    Right,
}

#[derive(Debug)]
pub struct CommonTangentsIntermediateData<T> {
    f_0: T,
    g_0: T,
    h_0: T,
    f_1: T,
    g_1: T,
    h_1: T,
    j: T,
    w: T,
    l: T,
    o: T,
    p: T,
    v: T,
    u: T,
    m: T,
}

impl<T> Ellipse<T> {
    pub fn center(&self) -> Point<T>
    where
        T: Clone,
    {
        (self.x.clone(), self.y.clone()).into()
    }

    pub fn axes(&self) -> Vector<T>
    where
        T: Clone,
    {
        (self.a.clone(), self.b.clone()).into()
    }

    pub fn rotation(&self) -> Complex<T>
    where
        T: Clone,
    {
        (self.r.clone(), self.i.clone()).into()
    }

    pub fn with_center(self, c: Point<T>) -> Self {
        let (x, y) = c.into();
        Self {
            x,
            y,
            a: self.a,
            b: self.b,
            r: self.r,
            i: self.i,
        }
    }

    pub fn with_axes(self, a: Vector<T>) -> Self {
        let (a, b) = a.into();
        Self {
            x: self.x,
            y: self.y,
            a,
            b,
            r: self.r,
            i: self.i,
        }
    }

    pub fn with_rotation(self, c: Complex<T>) -> Self {
        let (r, i) = c.into();
        Self {
            x: self.x,
            y: self.y,
            a: self.a,
            b: self.b,
            r,
            i,
        }
    }

    pub fn x(&self) -> &T {
        &self.x
    }
    pub fn y(&self) -> &T {
        &self.y
    }
    pub fn a(&self) -> &T {
        &self.a
    }
    pub fn b(&self) -> &T {
        &self.b
    }
    pub fn r(&self) -> &T {
        &self.r
    }
    pub fn i(&self) -> &T {
        &self.i
    }

    /// Into world space
    fn into_ws(&self, p: Point<T>) -> Point<T>
    where
        T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Clone + Zero,
    {
        (p.rotated(Point::origin(), self.rotation())).absolute(self.center())
    }

    /// From world space
    fn from_ws(&self, p: Point<T>) -> Point<T>
    where
        T: Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Neg<Output = T>
            + Sq<Output = NonNeg<T>>
            + Clone
            + Zero,
    {
        p.relative(self.center())
            .rotated(Point::origin(), !self.rotation())
    }

    pub fn from_raw(x: T, y: T, a: T, b: T, r: T, i: T) -> Self {
        Self { x, y, a, b, r, i }
    }

    pub fn new(center: Point<T>, axes: Vector<T>, rotation: Complex<T>) -> Self {
        let (x, y) = center.into();
        let (a, b) = axes.into();
        let (r, i) = rotation.into();
        Self { x, y, a, b, r, i }
    }

    pub fn from_angle(center: Point<T>, axes: Vector<T>, rotation: Angle<T>) -> Self
    where
        T: Sin<Output = T> + Cos<Output = T> + Clone,
    {
        let (x, y) = center.into();
        let (a, b) = axes.into();
        Self {
            x,
            y,
            a,
            b,
            r: rotation.clone().cos(),
            i: rotation.sin(),
        }
    }

    /// Focal point 0. Returns x, y of the point
    pub fn f0_old(&self) -> Point<T>
    where
        T: Abs<Output = NonNeg<T>>
            + PartialOrd
            + IsNan
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Sq<Output = NonNeg<T>>
            + Sqrt<Output = T>
            + Clone
            + Zero,
    {
        if self.a.clone().abs() > self.b.clone().abs() {
            let major_axis = absmax(self.a.clone(), self.b.clone());
            let minor_axis = absmin(self.a.clone(), self.b.clone());
            let focal_len = (major_axis.sq() - minor_axis.sq()).sqrt();

            self.into_ws((T::zero(), focal_len).into())
        } else {
            let major_axis = absmax(self.a.clone(), self.b.clone());
            let minor_axis = absmin(self.a.clone(), self.b.clone());
            let focal_len = (major_axis.sq() - minor_axis.sq()).sqrt();

            self.into_ws((focal_len, T::zero()).into())
        }
    }

    pub fn f0(&self) -> Point<T>
    where
        T: Abs<Output = NonNeg<T>>
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Sq<Output = NonNeg<T>>
            + PartialOrd
            + Clone
            + IsNeg
            + Zero,
        NonNeg<T>: Sqrt<Output = NonNeg<T>>,
    {
        if self.a.clone().abs() > self.b.clone().abs() {
            let focal_len = (NonNeg::new(self.a.clone().sq() - self.b.clone().sq())
                .ok()
                .unwrap())
            .sqrt();
            self.into_ws((T::zero(), focal_len.into_inner()).into())
        } else {
            let focal_len = (NonNeg::new(self.b.clone().sq() - self.a.clone().sq())
                .ok()
                .unwrap())
            .sqrt();
            self.into_ws((focal_len.into_inner(), T::zero()).into())
        }
    }

    /// Focal point 1. Returns x, y of the point
    pub fn f1(&self) -> Point<T>
    where
        T: Abs<Output = NonNeg<T>>
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Neg<Output = T>
            + Sq<Output = NonNeg<T>>
            + PartialOrd
            + IsNeg
            + IsNan
            + Clone
            + Zero,
        NonNeg<T>: Sqrt<Output = NonNeg<T>>,
    {
        if self.a.clone().abs() > self.b.clone().abs() {
            let major_axis = absmax(self.a.clone(), self.b.clone());
            let minor_axis = absmin(self.a.clone(), self.b.clone());
            let focal_len = NonNeg::new(major_axis.sq() - minor_axis.sq())
                .ok()
                .unwrap()
                .sqrt();
            self.into_ws((Zero::zero(), -focal_len.into_inner()).into())
        } else {
            let major_axis = absmax(self.a.clone(), self.b.clone());
            let minor_axis = absmin(self.a.clone(), self.b.clone());
            let focal_len = NonNeg::new(major_axis.sq() - minor_axis.sq())
                .ok()
                .unwrap()
                .sqrt();
            self.into_ws((-focal_len.into_inner(), Zero::zero()).into())
        }
    }

    pub fn from_foci(f0: Point<T>, f1: Point<T>, point_on_ellipse: Point<T>) -> Self
    where
        T: Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Sq<Output = NonNeg<T>>
            + Neg<Output = T>
            + Cos<Output = T>
            + Sin<Output = T>
            + IsNeg
            + One
            + Two
            + Clone
            + Pi,
        NonNeg<T>: Sqrt<Output = NonNeg<T>>,
    {
        let c = (f0.clone() - f1.clone()).len().into_inner() / T::two();

        // (sum / 2.)^2 = c^2 + b^2;
        // sum = (a-c)*2 + c*2
        // sum = a*2
        // a^2 == c^2 + b^2;

        let sum = (f0.clone() - point_on_ellipse.clone()).len().into_inner()
            + (f1.clone() - point_on_ellipse).len().into_inner();
        let a = sum / T::two();
        let b = NonNeg::new(a.clone().sq() - c.sq()).ok().unwrap().sqrt();

        let center = lerp(f0.clone(), f1, T::one() / T::two());

        let rot: Complex<T> = (f0 - center.clone()).rotor();

        let rot: Complex<T> =
            rot * Complex::from_polar(T::one(), Angle::from_radians(-T::pi() / T::two()));

        let (x, y) = center.into();
        let (r, i) = rot.into();
        Ellipse {
            x,
            y,
            a,
            b: b.into_inner(),
            r,
            i,
        }
    }

    pub fn radius(&self, anomaly: Angle<T>) -> T
    where
        T: Add<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Sq<Output = NonNeg<T>>
            + Cos<Output = T>
            + Sin<Output = T>
            + Clone,
        NonNeg<T>: Sqrt<Output = NonNeg<T>>,
    {
        let a = self.a.clone();
        let b = self.b.clone();
        a.clone() * b.clone()
            / ((b * anomaly.clone().cos()).sq() + (a * anomaly.sin()).sq())
                .sqrt()
                .into_inner()
    }

    pub fn apoapsis(&self) -> NonNeg<T>
    where
        T: Abs<Output = NonNeg<T>>
            + Add<Output = T>
            + Sub<Output = T>
            + PartialOrd
            + Sq<Output = NonNeg<T>>
            + Clone
            + IsNeg,
        NonNeg<T>: Sqrt<Output = NonNeg<T>>,
    {
        if self.a.clone().abs() > self.b.clone().abs() {
            let focal_len = NonNeg::new(self.a.clone().sq() - self.b.clone().sq())
                .ok()
                .unwrap()
                .sqrt();
            self.a.clone().abs() + focal_len
        } else {
            let focal_len = NonNeg::new(self.b.clone().sq() - self.a.clone().sq())
                .ok()
                .unwrap()
                .sqrt();
            self.b.clone().abs() + focal_len
        }
    }

    pub fn periapsis(&self) -> T
    where
        T: Abs<Output = NonNeg<T>>
            + Sub<Output = T>
            + Sq<Output = NonNeg<T>>
            + PartialOrd
            + Clone
            + IsNeg,
        NonNeg<T>: Sqrt<Output = NonNeg<T>>,
    {
        if self.a.clone().abs() > self.b.clone().abs() {
            let focal_len = NonNeg::new(self.a.clone().sq() - self.b.clone().sq())
                .ok()
                .unwrap()
                .sqrt();
            self.a.clone().abs() - focal_len
        } else {
            let focal_len = NonNeg::new(self.b.clone().sq() - self.a.clone().sq())
                .ok()
                .unwrap()
                .sqrt();
            self.b.clone().abs() - focal_len
        }
    }

    pub fn point_on_ellipse(&self, anomaly: Angle<T>) -> Point<T>
    where
        T: Abs<Output = NonNeg<T>>
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Cos<Output = T>
            + Sin<Output = T>
            + Clone
            + Zero,
    {
        self.into_ws(
            (
                self.b.clone().abs().into_inner() * anomaly.clone().sin(),
                self.a.clone().abs().into_inner() * anomaly.cos(),
            )
                .into(),
        )
    }

    pub fn acc(
        &self,
        anomaly: Angle<T>,
        central_body_mass: Kg<T>,
        gravitational_constant: T,
    ) -> Vector<T>
    where
        T: Cube<Output = T>
            + Abs<Output = NonNeg<T>>
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Cos<Output = T>
            + Sin<Output = T>
            + Sq<Output = NonNeg<T>>
            + PartialOrd
            + IsNeg
            + Clone
            + Zero,
        NonNeg<T>: Sqrt<Output = NonNeg<T>>,
    {
        let vec = self.f0() - self.point_on_ellipse(anomaly);
        let vec_len = vec.clone().len().into_inner();
        vec * central_body_mass.0 * gravitational_constant / vec_len.cube()
    }

    pub fn tangential_velocity(
        &self,
        anomaly: Angle<T>,
        central_body_mass: Kg<T>,
        gravitational_constant: T,
    ) -> Vector<T>
    where
        T: Abs<Output = NonNeg<T>>
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Cos<Output = T>
            + Sin<Output = T>
            + Neg<Output = T>
            + Sq<Output = NonNeg<T>>
            + PartialOrd
            + Clone
            + IsNeg
            + Zero,
        NonNeg<T>: Sqrt<Output = NonNeg<T>> + One + Two,
    {
        // let p = self.point_on_ellipse(t);
        // let center = (self.x, self.y);

        let vec = self.f0() - self.point_on_ellipse(anomaly.clone());
        let vec_len = vec.len();

        // let radius = self.radius(t);
        // println!("radius: {}", radius);
        let velocity_module = NonNeg::new(
            gravitational_constant
                * central_body_mass.0
                * (NonNeg::two() / vec_len - NonNeg::one() / self.a.clone().abs()),
        )
        .ok()
        .unwrap()
        .sqrt();

        // let tangent = (
        //     radius*T::sin(t * 2. * PI),
        //     radius*T::cos(t * 2. * PI),
        // );

        // let tangent = vmath::complex_mul( tangent, (self.i, self.r));

        let vel = Vector::from((
            // radius * T::sin(t * 2. * PI + PI/2.),
            // radius * T::cos(t * 2. * PI + PI/2.),
            self.b.clone().abs().into_inner() * anomaly.clone().cos(),
            self.a.clone().abs().into_inner() * -anomaly.sin(),
        )) * self.rotation();

        vel.norm() * velocity_module.into_inner()
    }

    pub fn angular_velocity(
        &self,
        anomaly: Angle<T>,
        central_body_mass: Kg<T>,
        gravitational_constant: T,
    ) -> DeltaAngle<T>
    where
        T: Abs<Output = NonNeg<T>>
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Cos<Output = T>
            + Sin<Output = T>
            + Neg<Output = T>
            + Sq<Output = NonNeg<T>>
            + PartialOrd
            + Clone
            + IsNeg
            + Zero,
        NonNeg<T>: Sqrt<Output = NonNeg<T>> + One + Two,
    {
        let p = self.point_on_ellipse(anomaly.clone());
        let f0 = self.f0();
        let r = (p - f0).len().into_inner();
        let v = self
            .tangential_velocity(anomaly, central_body_mass, gravitational_constant)
            .len()
            .into_inner();

        DeltaAngle::from_radians(v / r)
    }

    pub fn f1_from_tangential_velocity(
        &self,
        anomaly: Angle<T>,
        central_body_mass: Kg<T>,
        gravitational_constant: T,
        vel: Vector<T>,
    ) -> (Vector<T>, Point<T>)
    where
        T: Abs<Output = NonNeg<T>>
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Sq<Output = NonNeg<T>>
            + Cos<Output = T>
            + Sin<Output = T>
            + Neg<Output = T>
            + PartialOrd
            + Clone
            + IsNeg
            + Zero
            + One
            + Two,
        NonNeg<T>: Sqrt<Output = NonNeg<T>>,
    {
        let p = self.point_on_ellipse(anomaly);
        let f0 = self.f0();

        let _r = p - f0.clone();
        let _v = vel;
        let _mu = central_body_mass.0 * gravitational_constant;
        let _r_len = _r.clone().len().into_inner();
        let _v_len = _v.clone().len().into_inner();
        let _h = _r.clone().cross(_v.clone());
        let _energy = _v_len.clone().sq().into_inner() / T::two() - _mu.clone() / _r_len.clone();
        let _a = -_mu.clone() / (T::two() * _energy);
        let _e = (_r.clone() * (_v_len.sq().into_inner() - _mu.clone() / _r_len)
            - _v.clone() * _r.dot(_v))
            * (T::one() / _mu);
        let _f1 = _e.clone() * -T::two() * _a;

        (_e, f0 + _f1)
    }

    pub fn perimeter(&self) -> T
    where
        T: Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Sq<Output = NonNeg<T>>
            + IsNeg
            + Pi
            + One
            + UnsignedContstant<3>
            + UnsignedContstant<4>
            + UnsignedContstant<10>
            + Clone,
        NonNeg<T>: Sqrt<Output = NonNeg<T>>,
    {
        let h = (self.a.clone() - self.b.clone()).sq() / (self.a.clone() + self.b.clone()).sq();
        T::pi()
            * (self.a.clone() + self.b.clone())
            * (T::one()
                + uconst::<3, T>() * h.clone().into_inner()
                    / (uconst::<10, T>()
                        + NonNeg::new(uconst::<4, T>() - uconst::<3, T>() * h.into_inner())
                            .ok()
                            .unwrap()
                            .sqrt()
                            .into_inner()))
    }

    /// Change ellipse in the way that f0, and position in `t` stays the same and velocity in `t` changeds to `vel`
    pub fn set_tangential_velicity(
        &self,
        anomaly: Angle<T>,
        central_body_mass: Kg<T>,
        gravitational_constant: T,
        vel: Vector<T>,
    ) -> (T, T, T)
    where
        T: RemEuclid<Output = NonNeg<T>>
            + Abs<Output = NonNeg<T>>
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Cos<Output = T>
            + Sin<Output = T>
            + Neg<Output = T>
            + Sq<Output = NonNeg<T>>
            + PartialOrd
            + Clone
            + IsNeg
            + Zero
            + Pi
            + Two
            + One
            + UnsignedContstant<4>,
        NonNeg<T>: Sqrt<Output = NonNeg<T>>,
    {
        // consts
        let four = uconst::<4, T>();
        let p = self.point_on_ellipse(anomaly.clone());
        let f0 = self.f0();
        let vec_to_focus = f0.clone() - p.clone();
        #[allow(unused)]
        let vec_to_focus_len = vec_to_focus.len().into_inner();
        #[allow(unused)]
        let th = anomaly.radians().into_inner();
        #[allow(unused)]
        let velocity_module = vel.clone().len().into_inner();
        #[allow(unused)]
        let q = T::one()
            / (four.clone() / vec_to_focus_len
                - velocity_module.sq().into_inner() / gravitational_constant / central_body_mass.0);
        // ------

        // #[allow(unused)]
        // let (new_a, new_b, new_center, new_rotation): (T, T, Point<T>, Complex<T>) = todo!();

        /*

        #[allow(unreachable_code)]
        if new_a.abs() > new_b.abs() {
            assert!(
                new_center
                    == f0 - Vector::from((0., (q.sq() - new_b.sq()).sqrt())) * new_rotation
            );

            assert!(
                new_rotation
                    == Complex::div(
                        p - f0,
                        Vector::from((
                            new_b.abs() * T::sin(th),
                            q.abs() * T::cos(th) - (q.sq() - new_b.sq()).sqrt(),
                        ))
                    )
            ); // 3

            /*

                    {(a * c + b * d) / (c^2 + d^2) ,
                    (b * c - a * d) / (c^2 + d^2) }


                {
                    (new_b.abs() * T::cos(th) * new_b.abs() * T::sin(th) + q.abs() * -T::sin(th) * (q.abs() * T::cos(th) - (q.sq() - new_b.sq()).sqrt())) / ((new_b.abs() * T::sin(th))^2 + (q.abs() * T::cos(th) - (q.sq() - new_b.sq()).sqrt())^2) ,
                    (q.abs() * -T::sin(th) * new_b.abs() * T::sin(th) - new_b.abs() * T::cos(th) * (q.abs() * T::cos(th) - (q.sq() - new_b.sq()).sqrt())) / ((new_b.abs() * T::sin(th))^2 + (q.abs() * T::cos(th) - (q.sq() - new_b.sq()).sqrt())^2)
                }

                {
                    (
                          new_b^2 * cos(th) * sin(th)
                        - q^2 * cos(th) * sin(th)
                        + |q| * sin(th) * sqrt(q^2 - new_b^2)
                    )

                        / (
                            new_b^2 * sin(th)^2
                            - 2*|q| * cos(th) * sqrt(q^2 - new_b^2)
                            - new_b^2
                            + 2*q^2 * cos(th)^2
                        ) ,

                    (|q| * -sin(th) * |new_b| * sin(th) - |new_b| * cos(th) * (|q| * cos(th) - (q^2 - new_b^2).sqrt())) / ((|new_b| * sin(th))^2 + (|q| * cos(th) - (q^2 - new_b^2).sqrt())^2)
                }



                                    (
                          new_b^2 * cos(th) * sin(th)
                        - q^2 * cos(th) * sin(th)
                        + |q| * sin(th) * sqrt(q^2 - new_b^2)
                    )

                        / (
                            new_b^2 * sin(th)^2
                            - 2*|q| * cos(th) * sqrt(q^2 - new_b^2)
                            - new_b^2
                            + 2*q^2 * cos(th)^2
                        ) == (vel.x *(p.x-f0.x) + vel.y * (p.y-f0.y)) / ((p.x-f0.x)^2 + (p.y-f0.y)^2)





                         {(vel.x *(p.x-f0.x) + vel.y * (p.y-f0.y)) / ((p.x-f0.x)^2 + (p.y-f0.y)^2) ,
                    (vel.y * (p.x-f0.x) - vel.x * (p.y-f0.y)) / ((p.x-f0.x)^2 + (p.y-f0.y)^2) }

            */

            assert!(

                   com(
                         new_b.abs() * T::cos(th),
                         q.abs() * -T::sin(th)
                      )

                 / com(
                         new_b.abs() * T::sin(th),
                         q.abs() * T::cos(th) - (q.sq() - new_b.sq()).sqrt()
                      )

                == com(vel) / com(p - f0)

            );


        } else {
            assert!(self.into_ws(((new_b.sq() - q.sq()).sqrt(), 0.).into()) == f0);

            assert!(
                self.into_ws((new_b.abs() * T::sin(th), q.abs() * T::cos(th),).into()) == p
            );

            assert!(
                (Vector::from((new_b.abs() * T::cos(th), q.abs() * -T::sin(th),))
                    * new_rotation)
                    .norm()
                    == vel.norm()
            );
        }

        */

        let sin = |x| T::sin(x);
        let cos = |x| T::cos(x);

        let hx = (vel.x().clone() * (p.x().clone() - f0.x().clone())
            + vel.y().clone() * (p.y().clone() - f0.y().clone()))
            / ((p.x().clone() - f0.x().clone()).sq().into_inner()
                + (p.y().clone() - f0.y().clone()).sq().into_inner());
        let jx = hx.clone() * sin(th.clone()).sq().into_inner()
            - cos(th.clone()) * sin(th.clone())
            - hx.clone();
        let lx =
            hx.clone() * T::two() * q.clone().sq().into_inner() * cos(th.clone()).sq().into_inner()
                + q.clone().sq().into_inner() * cos(th.clone()) * sin(th.clone());
        let ox = (q.clone().abs().into_inner() * sin(th.clone())
            + hx * T::two() * q.clone().abs().into_inner() * cos(th))
        .sq();

        // let xxxx =
        //                jx^2 * new_b^4
        //              + (ox + 2 * jx * lx) * new_b^2
        //              - q^2 * ox - lx^2
        //              == 0

        let desc = NonNeg::new(
            (ox.clone().into_inner() + T::two() * jx.clone() * lx.clone())
                .sq()
                .into_inner()
                + four
                    * jx.clone().sq().into_inner()
                    * (q.clone().sq().into_inner() * ox.clone().into_inner()
                        + lx.clone().sq().into_inner()),
        )
        .ok()
        .unwrap();

        let new_b0 = (-(ox.clone().into_inner() + T::two() * jx.clone() * lx.clone())
            + desc.clone().sqrt().into_inner())
            / (T::two() * jx.clone().sq().into_inner());
        let new_b1 = (-(ox.into_inner() + T::two() * jx.clone() * lx) - desc.sqrt().into_inner())
            / (T::two() * jx.sq().into_inner());

        (q, new_b0, new_b1)
    }

    pub fn accelerated(
        &self,
        anomaly: Angle<T>,
        central_body_mass: Kg<T>,
        gravitational_constant: T,
        dt: Duration,
        acc: Vector<T>,
    ) -> Self
    where
        T: Abs<Output = NonNeg<T>>
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Cos<Output = T>
            + Sin<Output = T>
            + Neg<Output = T>
            + Sq<Output = NonNeg<T>>
            + PartialOrd
            + FromDurationAsSecs
            + Clone
            + IsNeg
            + Zero
            + One
            + Two
            + Pi,
        NonNeg<T>: Sqrt<Output = NonNeg<T>> + One + Two,
    {
        let f0 = self.f0();
        let p = self.point_on_ellipse(anomaly.clone());
        let vel = self.tangential_velocity(
            anomaly.clone(),
            central_body_mass.clone(),
            gravitational_constant.clone(),
        );
        let (_, new_f1) = self.f1_from_tangential_velocity(
            anomaly,
            central_body_mass,
            gravitational_constant,
            vel + acc * T::from_duration_as_secs(dt),
        );
        Ellipse::from_foci(f0, new_f1, p)
    }

    pub fn eccentricity(&self) -> NonNeg<T>
    where
        T: Sub<Output = T> + Div<Output = T> + Sq<Output = NonNeg<T>> + IsNeg + Clone + One,
        NonNeg<T>: Sqrt<Output = NonNeg<T>>,
    {
        NonNeg::new(T::one() - (self.b.clone() / self.a.clone()).sq().into_inner())
            .ok()
            .unwrap()
            .sqrt()
    }

    pub fn time_between_anomalies(
        &self,
        anomaly0: Angle<T>,
        anomaly1: Angle<T>,
        central_body_mass: Kg<T>,
        gravitational_constant: T,
    ) -> Duration
    where
        T: RemEuclid<Output = NonNeg<T>>
            + Cube<Output = T>
            + Atan<Output = T>
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Sin<Output = T>
            + Tan<Output = T>
            + Sq<Output = NonNeg<T>>
            + AddAssign
            + Clone
            + IsNeg
            + One
            + Two
            + Pi,
        NonNeg<T>: Sqrt<Output = NonNeg<T>>,
        Duration: FromSecs<T>,
    {
        time_between_true_anomalies(
            self.a.clone(),
            self.eccentricity(),
            anomaly0,
            anomaly1,
            central_body_mass.0 * gravitational_constant,
        )
    }

    /// Unlike `time_between_anomalies` can return negative time
    pub fn relative_time_between_anomalies(
        &self,
        anomaly0: Angle<T>,
        anomaly1: Angle<T>,
        central_body_mass: Kg<T>,
        gravitational_constant: T,
    ) -> RelativeDuration
    where
        T: RemEuclid<Output = NonNeg<T>>
            + Cube<Output = T>
            + Atan<Output = T>
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Sin<Output = T>
            + Tan<Output = T>
            + Sq<Output = NonNeg<T>>
            + IsNeg
            + Clone
            + One
            + Two
            + Pi,
        NonNeg<T>: Sqrt<Output = NonNeg<T>>,
        RelativeDuration: FromSecs<T>,
    {
        relative_time_between_true_anomalies(
            self.a.clone(),
            self.eccentricity(),
            anomaly0,
            anomaly1,
            central_body_mass.0 * gravitational_constant,
        )
    }

    pub(crate) fn eq_no_rot(&self) -> impl FnOnce(Point<T>) -> T
    where
        T: Add<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Sq<Output = NonNeg<T>>
            + Clone
            + One,
    {
        let x_0 = self.x.clone();
        let y_0 = self.y.clone();
        let a = self.a.clone();
        let b = self.b.clone();
        move |point| {
            let (x, y) = point.into();
            ((x - x_0.clone()).sq() / a.clone().sq() + (y - y_0.clone()).sq() / b.clone().sq())
                .into_inner()
                - T::one()
        }
    }

    pub fn eq(&self) -> impl FnOnce(Point<T>) -> T
    where
        T: Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Sq<Output = NonNeg<T>>
            + Clone
            + One,
    {
        let x_0 = self.x.clone();
        let y_0 = self.y.clone();
        let a = self.a.clone();
        let b = self.b.clone();
        let r = self.r.clone();
        let i = self.i.clone();
        move |point| {
            let (x, y) = point.into();
            ((r.clone() * (x.clone() - x_0.clone()) - i.clone() * (y.clone() - y_0.clone())).sq()
                / a.sq()
                + (r * (y - y_0) + i * (x - x_0)).sq() / b.sq())
            .into_inner()
                - T::one()
        }
    }

    /// (r*(x - x_0) - i*(y - y_0))^2 / a^2 + (r*(y - y_0) + i*(x - x_0))^2 / b^2 - 1
    /// y = k*x+d
    ///
    /// (r * (x - x_0) - i * (k * x + d - y_0))^2 / a^2 + (r * (k * x + d - y_0) + i * (x - x_0))^2 / b^2 - 1
    /// (r * (x - x_0) - e * (k * x + d - y_0))^2 / a^2 + (r * (k * x + d - y_0) + e * (x - x_0))^2 / b^2 - 1
    pub(crate) fn intersection_line_eq(&self, line: Line<T>) -> impl FnOnce(T) -> T
    where
        T: Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Sq<Output = NonNeg<T>>
            + Clone
            + One,
    {
        let x_0 = self.x.clone();
        let y_0 = self.y.clone();
        let a = self.a.clone();
        let b = self.b.clone();
        let r = self.r.clone();
        let i = self.i.clone();
        let k = line.k.clone();
        let d = line.d.clone();
        move |x| {
            ((r.clone() * (x.clone() - x_0.clone())
                - i.clone() * (k.clone() * x.clone() + d.clone() - y_0.clone()))
            .sq()
                / a.sq()
                + (r * (k * x.clone() + d - y_0) + i * (x - x_0)).sq() / b.sq())
            .into_inner()
                - T::one()
        }
    }

    /// returns discriminant of intersection equesion with `line`
    /// -(4 * (a^2 * (-k^2 * r^2 - 2 * i * k * r - i^2) + b^2 * (-i^2 * k^2 + 2 * i * k * r - r^2) + (r^4 + 2 * i^2 * r^2 + i^4) * (d^2 + 2 * d * (k * x_0 - y_0) + k^2 * x_0^2 - 2 * k * x_0 * y_0 + y_0^2))) / (a^2 * b^2)
    /// -(4 * (a_0^2 * (-k^2 * r_0^2 - 2 * i_0 * k * r_0 - i_0^2) + b_0^2 * (-i_0^2 * k^2 + 2 * i_0 * k * r_0 - r_0^2) + (r_0^4 + 2 * i_0^2 * r_0^2 + i_0^4) * (d^2 + 2 * d * (k * x_0 - y_0) + k^2 * x_0^2 - 2 * k * x_0 * y_0 + y_0^2))) / (a_0^2 * b_0^2)
    /// -(4 * (a_1^2 * (-k^2 * r_1^2 - 2 * i_1 * k * r_1 - i_1^2) + b_1^2 * (-i_1^2 * k^2 + 2 * i_1 * k * r_1 - r_1^2) + (r_1^4 + 2 * i_1^2 * r_1^2 + i_1^4) * (d^2 + 2 * d * (k * x_1 - y_1) + k^2 * x_1^2 - 2 * k * x_1 * y_1 + y_1^2))) / (a_1^2 * b_1^2)
    pub fn intersection_discriminant(&self, line: Line<T>) -> T
    where
        T: Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Sq<Output = NonNeg<T>>
            + Clone
            + One
            + Two
            + UnsignedContstant<4>,
    {
        let x_0 = self.x.clone();
        let y_0 = self.y.clone();
        let a = self.a.clone();
        let b = self.b.clone();
        let r = self.r.clone();
        let i = self.i.clone();
        let k = line.k.clone();
        let d = line.d.clone();

        uconst::<4, T>()
            * ((a.clone() * (r.clone() * k.clone() + i.clone())).sq()
                + (b.clone() * (i * k.clone() - r)).sq()
                - (d.clone() + k.clone() * x_0.clone()).sq()
                + T::two() * d * y_0.clone()
                + T::two() * k * x_0 * y_0.clone()
                - y_0.sq().into_inner())
            / (a * b).sq().into_inner()
    }

    /// returns `d` by given `k` where `y = kx + d` is a tangent to ellipse
    pub fn tangent_d(&self, k: T) -> (T, T)
    where
        T: Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Neg<Output = T>
            + Sq<Output = NonNeg<T>>
            + Clone,
        NonNeg<T>: Sqrt<Output = NonNeg<T>>,
    {
        let x_0 = self.x.clone();
        let y_0 = self.y.clone();
        let a = self.a.clone();
        let b = self.b.clone();
        let r = self.r.clone();
        let i = self.i.clone();

        let discriminant =
            (a * (r.clone() * k.clone() + i.clone())).sq() + (b * (i * k.clone() - r)).sq();
        let base = -k * x_0 + y_0;

        (
            base.clone() + discriminant.clone().sqrt().into_inner(),
            base - discriminant.sqrt().into_inner(),
        )
    }

    /// intersection of this function with y = 0 is where common outer tangents are
    pub fn outer_tangents_fun(&self, rhs: &Self, k: T) -> (T, T)
    where
        T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Sq<Output = NonNeg<T>> + Clone,
        NonNeg<T>: Sqrt<Output = NonNeg<T>>,
    {
        let x_0 = self.x.clone();
        let y_0 = self.y.clone();
        let a_0 = self.a.clone();
        let b_0 = self.b.clone();
        let r_0 = self.r.clone();
        let i_0 = self.i.clone();

        let x_1 = rhs.x.clone();
        let y_1 = rhs.y.clone();
        let a_1 = rhs.a.clone();
        let b_1 = rhs.b.clone();
        let r_1 = rhs.r.clone();
        let i_1 = rhs.i.clone();

        let discriminant_0 = (a_0 * (r_0.clone() * k.clone() + i_0.clone())).sq()
            + (b_0 * (i_0 * k.clone() - r_0)).sq();
        let discriminant_1 = (a_1 * (r_1.clone() * k.clone() + i_1.clone())).sq()
            + (b_1 * (i_1 * k.clone() - r_1)).sq();

        let lhs = k * (x_1 - x_0) + y_0 - y_1;
        let rhs = discriminant_1.sqrt() - discriminant_0.sqrt();

        (lhs.clone() - rhs.clone(), lhs + rhs)
    }

    pub fn common_tangents_intermediate_data(&self, rhs: &Self) -> CommonTangentsIntermediateData<T>
    where
        T: SignedSq<Output = T>
            + SignedSqrt<Output = T>
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Abs<Output = NonNeg<T>>
            + Pow<T, Output = T>
            + Sq<Output = NonNeg<T>>
            + IsPositive
            + DivAssign
            + FromUSize
            + Clone
            + Zero
            + One
            + Two
            + Two
            + Sum
            + UnsignedContstant<4>,
        Positive<T>: Log2<Output = T>,
        NonNeg<T>: One,
    {
        let x_0 = self.x.clone();
        let y_0 = self.y.clone();
        let a_0 = self.a.clone();
        let b_0 = self.b.clone();
        let r_0 = self.r.clone();
        let i_0 = self.i.clone();

        let x_1 = rhs.x.clone();
        let y_1 = rhs.y.clone();
        let a_1 = rhs.a.clone();
        let b_1 = rhs.b.clone();
        let r_1 = rhs.r.clone();
        let i_1 = rhs.i.clone();

        let c = NonNeg::<T>::one();

        let f_0 = ((a_0.clone() * r_0.clone()).sq() + (b_0.clone() * i_0.clone()).sq()) / c.clone();
        let g_0 = (T::two() * i_0.clone() * r_0.clone() * (a_0.clone().sq() - b_0.clone().sq()))
            / c.clone().into_inner();
        let h_0 = ((a_0 * i_0).sq() + (b_0 * r_0).sq()) / c.clone();

        let f_1 = ((a_1.clone() * r_1.clone()).sq() + (b_1.clone() * i_1.clone()).sq()) / c.clone();
        let g_1 = (T::two() * i_1.clone() * r_1.clone() * (a_1.clone().sq() - b_1.clone().sq()))
            / c.clone().into_inner();
        let h_1 = ((a_1 * i_1).sq() + (b_1 * r_1).sq()) / c.clone();

        let dx = (x_1 - x_0).ssq() / c.clone().into_inner();
        let dy = (y_1 - y_0).ssq() / c.into_inner();

        // println!(
        //     "before: {}, {}, {}, {}, {}, {}, {}, {}",
        //     f_0, g_0, h_0, f_1, g_1, h_1, dx, dy
        // );
        let [f_0, g_0, h_0, f_1, g_1, h_1, dx, dy] = notmalize_array_around_one([
            f_0.into_inner(),
            g_0,
            h_0.into_inner(),
            f_1.into_inner(),
            g_1,
            h_1.into_inner(),
            dx,
            dy,
        ]);
        // println!(
        //     "after : {}, {}, {}, {}, {}, {}, {}, {}",
        //     f_0, g_0, h_0, f_1, g_1, h_1, dx, dy
        // );

        let dx = dx.ssqrt();
        let dy = dy.ssqrt();

        let j = f_1.clone() + f_0.clone() - dx.clone().sq().into_inner();
        let w = g_1.clone() + g_0.clone() + T::two() * dx * dy.clone();
        let l = h_1.clone() + h_0.clone() - dy.sq().into_inner();

        let o = j.clone().sq().into_inner() - uconst::<4, T>() * f_1.clone() * f_0.clone();
        let p = T::two() * j.clone() * w.clone()
            - uconst::<4, T>() * f_1.clone() * g_0.clone()
            - uconst::<4, T>() * f_0.clone() * g_1.clone();
        let v = w.clone().sq().into_inner() + T::two() * j.clone() * l.clone()
            - uconst::<4, T>() * f_1.clone() * h_0.clone()
            - uconst::<4, T>() * g_1.clone() * g_0.clone()
            - uconst::<4, T>() * h_1.clone() * f_0.clone();
        let u = T::two() * w.clone() * l.clone()
            - uconst::<4, T>() * g_1.clone() * h_0.clone()
            - uconst::<4, T>() * h_1.clone() * g_0.clone();
        let m = l.clone().sq().into_inner() - uconst::<4, T>() * h_1.clone() * h_0.clone();

        CommonTangentsIntermediateData {
            f_0,
            g_0,
            h_0,
            f_1,
            g_1,
            h_1,
            j,
            w,
            l,
            o,
            p,
            v,
            u,
            m,
        }
    }

    pub fn tangent_k_alg(&self, rhs: &Self, k: T) -> (T, T)
    where
        T: Cube<Output = T>
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Abs<Output = NonNeg<T>>
            + Pow<T, Output = T>
            + Sq<Output = NonNeg<T>>
            + TopLimit
            + Clone
            + IsNeg
            + Zero
            + Two
            + UnsignedContstant<4>
            + UnsignedContstant<100>
            + UnsignedContstant<20000000>,
        NonNeg<T>: Sqrt<Output = NonNeg<T>>,
    {
        let x_0 = self.x.clone();
        let y_0 = self.y.clone();
        let a_0 = self.a.clone();
        let b_0 = self.b.clone();
        let r_0 = self.r.clone();
        let i_0 = self.i.clone();

        let x_1 = rhs.x.clone();
        let y_1 = rhs.y.clone();
        let a_1 = rhs.a.clone();
        let b_1 = rhs.b.clone();
        let r_1 = rhs.r.clone();
        let i_1 = rhs.i.clone();

        let eq = |left: T, right: T| (left - right).abs();

        let f_0 = (a_0.clone() * r_0.clone()).sq() + (b_0.clone() * i_0.clone()).sq();
        let g_0 = T::two() * i_0.clone() * r_0.clone() * (a_0.clone().sq() - b_0.clone().sq());
        let h_0 = (a_0.clone() * i_0.clone()).sq() + (b_0.clone() * r_0.clone()).sq();

        let _discriminant_0 = (k.clone().sq() * f_0.clone()).into_inner()
            + k.clone() * g_0.clone()
            + h_0.clone().into_inner();

        //let discriminant_0
        //    = (a_0 * (r_0 * k + i_0)).sq()
        //    + (b_0 * (i_0 * k - r_0)).sq();

        // (a^2 + b^2).sqrt() - (c^2 + d^2).sqrt() = z;
        // (a^2 + b^2) + (c^2 + d^2) - (a^2 + b^2).sqrt() * (c^2 + d^2).sqrt() = z^2
        // (a^2 + b^2 + c^2 + d^2 - z^2)^2 = (a^2 + b^2)(c^2 + d^2)
        //

        let f_1 = (a_1.clone() * r_1.clone()).sq() + (b_1.clone() * i_1.clone()).sq();
        let g_1 = T::two() * i_1.clone() * r_1.clone() * (a_1.clone().sq() - b_1.clone().sq());
        let h_1 = (a_1.clone() * i_1.clone()).sq() + (b_1.clone() * r_1.clone()).sq();

        let _discriminant_1 = (k.clone().sq() * f_1.clone()).into_inner()
            + k.clone() * g_1.clone()
            + h_1.clone().into_inner();

        //let discriminant_1
        //    = (a_1 * (r_1 * k + i_1)).sq()
        //    + (b_1 * (i_1 * k - r_1)).sq();

        let _rhs = NonNeg::new(
            (k.clone().sq() * f_1.clone()).into_inner()
                + k.clone() * g_1.clone()
                + h_1.clone().into_inner(),
        )
        .ok()
        .unwrap()
        .sqrt()
            - NonNeg::new(
                (k.clone().sq() * f_0.clone()).into_inner()
                    + k.clone() * g_0.clone()
                    + h_0.clone().into_inner(),
            )
            .ok()
            .unwrap()
            .sqrt();

        let _discriminant_0 = (a_0.clone() * (r_0.clone() * k.clone() + i_0.clone())).sq()
            + (b_0.clone() * (i_0.clone() * k.clone() - r_0.clone())).sq();
        let _discriminant_1 = (a_1.clone() * (r_1.clone() * k.clone() + i_1.clone())).sq()
            + (b_1.clone() * (i_1.clone() * k.clone() - r_1.clone())).sq();

        //= a.sq()
        //+ b.sq()
        //- 2. * a * b
        //- 2. * a * c.sq()
        //- 2. * b * c.sq()
        //+ c.pow(4.)

        let eq0 = eq(
            (k.clone() * (x_1.clone() - x_0.clone()) + y_0.clone() - y_1.clone())
                .sq()
                .into_inner(),
            (NonNeg::new((k.clone().sq() * f_1).into_inner() + k.clone() * g_1 + h_1.into_inner())
                .ok()
                .unwrap()
                .sqrt()
                - NonNeg::new(
                    (k.clone().sq() * f_0).into_inner() + k.clone() * g_0 + h_0.into_inner(),
                )
                .ok()
                .unwrap()
                .sqrt())
            .sq()
            .into_inner(),
        );

        let f_0 = (a_0.clone() * r_0.clone()).sq() + (b_0.clone() * i_0.clone()).sq();
        let g_0 = T::two() * i_0.clone() * r_0.clone() * (a_0.clone().sq() - b_0.clone().sq());
        let h_0 = (a_0 * i_0).sq() + (b_0 * r_0).sq();

        let f_1 = (a_1.clone() * r_1.clone()).sq() + (b_1.clone() * i_1.clone()).sq();
        let g_1 = T::two() * i_1.clone() * r_1.clone() * (a_1.clone().sq() - b_1.clone().sq());
        let h_1 = (a_1 * i_1).sq() + (b_1 * r_1).sq();

        let j = f_1.clone() + f_0.clone() - (x_0.clone() - x_1.clone()).sq();
        let w = g_1.clone() + g_0.clone() - T::two() * (x_0 - x_1) * (y_1.clone() - y_0.clone());
        let l = h_1.clone() + h_0.clone() - (y_1 - y_0).sq();

        let eq1 = if !(k.clone().sq().into_inner() * j.clone() + k.clone() * w.clone() + l.clone())
            .is_neg()
        {
            let o = j.clone().sq().into_inner()
                - uconst::<4, T>() * f_1.clone().into_inner() * f_0.clone().into_inner();
            let p = T::two() * j.clone() * w.clone()
                - uconst::<4, T>() * f_1.clone().into_inner() * g_0.clone()
                - uconst::<4, T>() * f_0.clone().into_inner() * g_1.clone();
            let v = T::two() * j * l.clone() + w.clone().sq().into_inner()
                - uconst::<4, T>() * f_1.into_inner() * h_0.clone().into_inner()
                - uconst::<4, T>() * g_1.clone() * g_0.clone()
                - uconst::<4, T>() * h_1.clone().into_inner() * f_0.into_inner();
            let u = T::two() * w * l.clone()
                - uconst::<4, T>() * g_1 * h_0.clone().into_inner()
                - uconst::<4, T>() * h_1.clone().into_inner() * g_0;
            let m = l.sq().into_inner() - uconst::<4, T>() * h_1.into_inner() * h_0.into_inner();

            let final_val = k.clone().pow(uconst::<4, T>()) * o
                + k.clone().cube() * p
                + k.clone().sq().into_inner() * v
                + k * u
                + m;

            eq(final_val, T::zero()).into_inner()
        } else {
            T::top_limit()
        };

        // let eq0 = eq(k * (x_1 - x_0) + y_0 - y_1, rhs);

        // let eq1 = eq(k * (x_1 - x_0) + y_0 - y_1, -rhs);

        // let eq0 = eq(
        //     k * (x_1 - x_0) + y_0 - y_1,
        //     discriminant_1.sqrt() - discriminant_0.sqrt(),
        // );

        // let eq1 = eq(
        //     k * (x_0 - x_1) + y_1 - y_0,
        //     discriminant_1.sqrt() - discriminant_0.sqrt(),
        // );

        (
            eq0.into_inner() / uconst::<100, T>(),
            eq1 / uconst::<20000000, T>(),
        )
    }
}

impl Ellipse<f64> {
    pub fn common_tangents(&self, rhs: &Self) -> Vec<(Line<f64>, TangentDirection)> {
        let id = self.common_tangents_intermediate_data(rhs);

        fn pp<'a, T>(
            e0: &'a Ellipse<T>,
            e1: &'a Ellipse<T>,
            j: T,
            w: T,
            l: T,
            roots: &'a [T],
            err: NonNeg<T>,
        ) -> impl Iterator<Item = (Line<T>, TangentDirection)> + 'a
        where
            T: Add<Output = T>
                + Sub<Output = T>
                + Mul<Output = T>
                + Neg<Output = T>
                + Abs<Output = NonNeg<T>>
                + Sq<Output = NonNeg<T>>
                + IsNeg
                + Clone
                + PartialOrd,
            NonNeg<T>: Sqrt<Output = NonNeg<T>>,
        {
            //println!("roots: {:?}", roots);
            roots
                .into_iter()
                .filter(move |k| {
                    !((*k).clone().sq().into_inner() * j.clone()
                        + (*k).clone() * w.clone()
                        + l.clone())
                    .is_neg()
                })
                .map(move |k| {
                    let d_0 = e0.tangent_d(k.clone());
                    let d_1 = e1.tangent_d(k.clone());

                    let mut vec = Vec::new();

                    if (d_0.0.clone() - d_1.0.clone()).abs() < err
                        || (d_0.0.clone() - d_1.1.clone()).abs() < err
                    {
                        vec.push((
                            Line {
                                k: k.clone(),
                                d: d_0.0,
                            },
                            TangentDirection::Left,
                        ));
                    }
                    if (d_0.1.clone() - d_1.0).abs() < err || (d_0.1.clone() - d_1.1).abs() < err {
                        vec.push((
                            Line {
                                k: k.clone(),
                                d: d_0.1,
                            },
                            TangentDirection::Right,
                        ));
                    }
                    vec
                })
                .flatten()
        }

        //println!("pol: {}, {}, {}, {}, {}", o, p, v, u, m);
        //let norm = notmalize_array([id.o, id.p, id.v, id.u, id.m]);
        //println!("norm: {:?}", norm);

        let poly = Polynomial::<f64>::new(vec![id.o, id.p, id.v, id.u, id.m]);

        let err = non_neg!(0.1);
        //println!("roots2: {:?}", );

        let res = match poly.roots() {
            rustnomial::Roots::NoRoots => pp(self, rhs, id.j, id.w, id.l, &[], err).collect(),
            rustnomial::Roots::NoRootsFound => pp(self, rhs, id.j, id.w, id.l, &[], err).collect(),
            rustnomial::Roots::OneRealRoot(root) => {
                pp(self, rhs, id.j, id.w, id.l, &[root], err).collect()
            }
            rustnomial::Roots::TwoRealRoots(r0, r1) => {
                pp(self, rhs, id.j, id.w, id.l, &[r0, r1], err).collect()
            }
            rustnomial::Roots::ThreeRealRoots(r0, r1, r2) => {
                pp(self, rhs, id.j, id.w, id.l, &[r0, r1, r2], err).collect()
            }
            rustnomial::Roots::ManyRealRoots(roots) => pp(
                self,
                rhs,
                id.j,
                id.w,
                id.l,
                &roots.iter().map(|x| *x).collect::<Vec<_>>(),
                err,
            )
            .collect(),
            rustnomial::Roots::OneComplexRoot(_) => {
                pp(self, rhs, id.j, id.w, id.l, &[], err).collect()
            }
            rustnomial::Roots::TwoComplexRoots(_, _) => {
                pp(self, rhs, id.j, id.w, id.l, &[], err).collect()
            }
            rustnomial::Roots::ThreeComplexRoots(_, _, _) => {
                pp(self, rhs, id.j, id.w, id.l, &[], err).collect()
            }
            rustnomial::Roots::ManyComplexRoots(_) => {
                pp(self, rhs, id.j, id.w, id.l, &[], err).collect()
            }
            rustnomial::Roots::InfiniteRoots => pp(self, rhs, id.j, id.w, id.l, &[], err).collect(),
            rustnomial::Roots::OnlyRealRoots(roots) => pp(
                self,
                rhs,
                id.j,
                id.w,
                id.l,
                &roots.iter().map(|x| *x).collect::<Vec<_>>(),
                err,
            )
            .collect(),
        };

        // let res = match roots::find_roots_quartic(id.o, id.p, id.v, id.u, id.m) {
        //     roots::Roots::No(roots) => pp(self, rhs, id.j, id.w, id.l, roots).collect(),
        //     roots::Roots::One(roots) => pp(self, rhs, id.j, id.w, id.l, roots).collect(),
        //     roots::Roots::Two(roots) => pp(self, rhs, id.j, id.w, id.l, roots).collect(),
        //     roots::Roots::Three(roots) => pp(self, rhs, id.j, id.w, id.l, roots).collect(),
        //     roots::Roots::Four(roots) => pp(self, rhs, id.j, id.w, id.l, roots).collect(),
        // };

        // for r in &res {
        //     println!("res: {:?}", r);
        // }

        res

        // let eq1 = if  {

        //     let final_val = k.pow(4.) * o + k.pow(3.) * p + k.sq() * v + k * u + m;

        //     eq(final_val, 0.)
        // } else {
        //     T::MAX
        // };

        //(eq0 / 100., eq1 / 20000000.)
    }
}

fn time_between_true_anomalies<T>(
    major_axis: T,
    eccentricity: NonNeg<T>,
    nu1: Angle<T>,
    nu2: Angle<T>,
    mu: T,
) -> Duration
where
    T: RemEuclid<Output = NonNeg<T>>
        + Cube<Output = T>
        + Atan<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Sin<Output = T>
        + Tan<Output = T>
        + AddAssign
        + IsNeg
        + One
        + Two
        + Pi
        + Clone,
    NonNeg<T>: Sqrt<Output = NonNeg<T>>,
    Duration: FromSecs<T>,
{
    let e1 = true_anomaly_to_eccentric(eccentricity.clone(), nu1);
    let e2 = true_anomaly_to_eccentric(eccentricity.clone(), nu2);

    let m1 = eccentric_anomaly_to_mean(eccentricity.clone(), e1);
    let m2 = eccentric_anomaly_to_mean(eccentricity.clone(), e2);

    // Mean motion (n = sqrt(mu / a^3))
    let n = NonNeg::new(mu / major_axis.cube()).ok().unwrap().sqrt();

    let mut delta_m = m2 - m1;

    // Ensure the time is positive if traveling forward
    if delta_m.is_neg() {
        delta_m += T::two() * T::pi();
    }

    <Duration as FromSecs<T>>::from_secs(delta_m / n.into_inner())
}

fn relative_time_between_true_anomalies<T>(
    major_axis: T,
    eccentricity: NonNeg<T>,
    nu1: Angle<T>,
    nu2: Angle<T>,
    mu: T,
) -> RelativeDuration
where
    T: RemEuclid<Output = NonNeg<T>>
        + Cube<Output = T>
        + Atan<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Sin<Output = T>
        + Tan<Output = T>
        + IsNeg
        + Clone
        + One
        + Two
        + Pi,
    NonNeg<T>: Sqrt<Output = NonNeg<T>>,
    RelativeDuration: FromSecs<T>,
{
    let e1 = true_anomaly_to_eccentric(eccentricity.clone(), nu1);
    let e2 = true_anomaly_to_eccentric(eccentricity.clone(), nu2);

    let m1 = eccentric_anomaly_to_mean(eccentricity.clone(), e1);
    let m2 = eccentric_anomaly_to_mean(eccentricity, e2);

    // Mean motion (n = sqrt(mu / a^3))
    let n = NonNeg::new(mu / major_axis.cube()).ok().unwrap().sqrt();

    <RelativeDuration as FromSecs<T>>::from_secs((m2 - m1) / n.into_inner())
}

fn true_anomaly_to_eccentric<T>(eccentricity: NonNeg<T>, anomaly: Angle<T>) -> T
where
    T: RemEuclid<Output = NonNeg<T>>
        + Atan<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Tan<Output = T>
        + IsNeg
        + Clone
        + One
        + Two
        + Pi,
    NonNeg<T>: Sqrt<Output = NonNeg<T>>,
{
    let factor = NonNeg::new(
        (T::one() - eccentricity.clone().into_inner()) / (T::one() + eccentricity.into_inner()),
    )
    .ok()
    .unwrap()
    .sqrt();
    T::two() * (factor.into_inner() * (anomaly.radians().into_inner() / T::two()).tan()).atan()
}

fn eccentric_anomaly_to_mean<T>(eccentricity: NonNeg<T>, e_anom: T) -> T
where
    T: Sub<Output = T> + Mul<Output = T> + Sin<Output = T> + Clone,
{
    e_anom.clone() - eccentricity.into_inner() * e_anom.sin()
}

#[cfg(test)]
mod tests {
    use std::ops::Sub;

    use super::Ellipse;
    use burbomath::{Abs, NonNeg, Sq as _};
    use num_traits::Pow as _;

    macro_rules! assert_eq_err {
        ($x: expr, $y: expr, $err: expr) => {
            let x = $x;
            let y = $y;
            let err = $err;

            let d = (x - y).abs();
            if d > err {
                panic!(
                    "{} and {} have difference equal {} which exceeds {}",
                    x, y, d, err
                );
            }
        };
    }

    macro_rules! assert_ne_err {
        ($x: expr, $y: expr, $err: expr) => {
            let x = $x;
            let y = $y;
            let err = $err;

            let d = (x - y).abs();
            if d < err {
                panic!(
                    "{} and {} have difference equal {} which less then {}",
                    x, y, d, err
                );
            }
        };
    }

    static E0: Ellipse<f32> = Ellipse {
        x: 100.0,
        y: 100.0,
        a: 40.0,
        b: 70.0,
        r: 0.9659258,
        i: 0.25881904,
    };

    static E1: Ellipse<f32> = Ellipse {
        x: -30.0,
        y: -100.0,
        a: 20.0,
        b: 80.0,
        r: 0.50000036,
        i: -0.8660252,
    };

    // #[test]
    // fn tangent_d_d_0() {
    //     let k = deg_to_rad(66.).tan();
    //     let r = E0.tangent_k_alg(&E1, k);
    //     assert_eq_err!(r.0, 373., 2.);
    //     assert_eq_err!(r.1, 5., 2.);
    // }

    // #[test]
    // fn tangent_d_d_1() {
    //     let k = deg_to_rad(50.).tan();
    //     let r = E0.tangent_k_alg(&E1, k);
    //     assert_eq_err!(r.0, 9., 2.);
    //     assert_eq_err!(r.1, 189., 2.);
    // }

    // #[test]
    // fn tangent_d_d_2() {
    //     let k = deg_to_rad(85.).tan();
    //     let r = E0.tangent_k_alg(&E1, k);
    //     assert_eq_err!(r.0, 3300., 2.);
    //     assert_eq_err!(r.1, 1842., 2.);
    // }

    // #[test]
    // fn tangent_d_d_3() {
    //     let k = deg_to_rad(25.).tan();
    //     let r = E0.tangent_k_alg(&E1, k);
    //     assert_eq_err!(r.0, 263., 2.);
    //     assert_eq_err!(r.1, 294., 2.);
    // }

    fn eq<T>(left: T, right: T) -> NonNeg<T>
    where
        T: Sub<Output = T> + Abs<Output = NonNeg<T>>,
    {
        (left - right).abs()
    }

    #[test]
    fn xxx() {
        let fun0 = |a: f32, b: f32, c: f32| eq(c, a.sqrt() - b.sqrt());

        let fun1 = |a: f32, b: f32, c: f32| eq(a.sqrt() + b.sqrt(), (a - b) / c);

        let fun2 = |a: f32, b: f32, c: f32| {
            let q = (a - b) / c;
            // a = 9
            // b = 16
            // c = 1
            // q = -7

            // 7^2 == 9 + 16 + 2*3*4

            eq(q, a.sqrt() + b.sqrt())
            //eq(q.sq(), a + 2. * a.sqrt() * b.sqrt() + b)

            //eq(q.pow(4.) + 2. * a * q.sq() - 2. * b * q.sq() + a.sq() + b.sq() - 6. * a * b, 0.)
        };

        // q = (a - b) / c

        // q = a.sqrt() + b.sqrt();
        // q.sq() = a + 2 * a.sqrt() * b.sqrt() + b;
        // q.sq() - a - b = 2 * a.sqrt() * b.sqrt()
        // (q.sq() + (a - b)).sq() = 4 * a * b
        // q.pow(4.) + 2 * c.sq() * (a - b) + (a - b)^2 = 4 * a * b
        // c.pow(4.) + 2 * a * c.sq() - 2 * b * c.sq() + a.sq() - 2*a*b + b.sq() - 4*a*b = 0
        // c.pow(4.) + 2 * a * c.sq() - 2 * b * c.sq() + a.sq() + b.sq() - 6*a*b = 0

        let fun3 = |a: f32, b: f32, c: f32| {
            eq(
                a.sq().into_inner() + b.sq().into_inner()
                    - 2. * a * b
                    - 2. * a * c.sq().into_inner()
                    - 2. * b * c.sq().into_inner()
                    + c.pow(4.),
                0.,
            )
        };

        //c = a.sqrt() - b.sqrt();
        //c^2 = a + b - 2 * a.sqrt() * b.sqrt()
        //c^2 - a - b = - 2 * a.sqrt() * b.sqrt()
        //a + b - c^2 = 2 * a.sqrt() * b.sqrt()
        //(a + b)^2 - 2(a + b)c^2 + c^4 = 4ab
        //a^2 + b^2 + 2ab - 2ac^2 - 2bc^2 + c^4 = 4ab
        //a^2 + b^2 - 2ab - 2ac^2 - 2bc^2 + c^4 = 0
        let do_assert = |fun: fn(f32, f32, f32) -> NonNeg<f32>| {
            assert_eq_err!(fun(9., 4., 1.).into_inner(), 0., 0.001);
            assert_eq_err!(fun(16., 4., 2.).into_inner(), 0., 0.001);
            assert_eq_err!(fun(16., 9., 1.).into_inner(), 0., 0.001);
            assert_eq_err!(fun(9., 16., -1.).into_inner(), 0., 0.001);
            assert_ne_err!(fun(9., 16., 1.).into_inner(), 0., 0.001);
        };
        do_assert(fun0);
        do_assert(fun1);
        do_assert(fun2);
        //do_assert(fun3);
    }

    // #[test]
    // fn rots() {
    //     let den = 100000.;
    //     //530586800 * x^4 + (-3043730000) * x^3 + 5108047400 * x^2 + (-3046621000) * x + 502631040 = 0;
    //     let roots = roots::find_roots_quartic(
    //         530586800_T / den,
    //         -3043730000_T / den,
    //         5108047400_T / den,
    //         -3046621000_T / den,
    //         502631040_T / den,
    //     );
    //     assert_eq!(roots, Roots::Four([0.26496, 0.81798, 1.30545, 3.34813]))
    // }
}
