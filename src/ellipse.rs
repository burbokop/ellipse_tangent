use burbomath::{
    Angle, Complex, DeltaAngle, Point, SignedSq as _, SignedSqrt as _, Sq as _, Vector, lerp,
    physics::Kg,
};
use num_traits::Pow as _;
use rustnomial::Polynomial;
use std::{f32::consts::PI, time::Duration};

use crate::{
    line::Line,
    utils::{RelativeDuration, notmalize_array_around_one},
};

#[inline(always)]
fn absmax(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return f32::NAN;
    }

    if a.abs() > b.abs() { a } else { b }
}

#[inline(always)]
fn absmin(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return f32::NAN;
    }

    if a.abs() < b.abs() { a } else { b }
}

// pub mod vmath {
//     use num_traits::Pow as _;

//     #[inline(always)]
//     pub fn complex_mul(ab: (f32, f32), cd: (f32, f32)) -> (f32, f32) {
//         let (a, b) = ab;
//         let (c, d) = cd;
//         (a * c - b * d, a * d + b * c)
//     }

//     pub fn add(vec0: (f32, f32), vec1: (f32, f32)) -> (f32, f32) {
//         let (x0, y0) = vec0;
//         let (x1, y1) = vec1;
//         (x0 + x1, y0 + y1)
//     }

//     pub fn sub(vec0: (f32, f32), vec1: (f32, f32)) -> (f32, f32) {
//         let (x0, y0) = vec0;
//         let (x1, y1) = vec1;
//         (x0 - x1, y0 - y1)
//     }

//     pub fn left_perp(vec: (f32, f32)) -> (f32, f32) {
//         let (x, y) = vec;
//         (-y, x)
//     }

//     pub fn right_perp(vec: (f32, f32)) -> (f32, f32) {
//         let (x, y) = vec;
//         (y, -x)
//     }

//     pub fn len(vec: (f32, f32)) -> f32 {
//         let (x, y) = vec;
//         (x.pow(2.) + y.pow(2.)).sqrt()
//     }

//     pub fn norm(vec: (f32, f32)) -> (f32, f32) {
//         div(vec, len(vec))
//     }

//     pub fn mul(vec: (f32, f32), s: f32) -> (f32, f32) {
//         let (x, y) = vec;
//         (x * s, y * s)
//     }

//     pub fn div(vec: (f32, f32), s: f32) -> (f32, f32) {
//         let (x, y) = vec;
//         (x / s, y / s)
//     }
// }

#[derive(Debug, Clone, PartialEq)]
pub struct Ellipse {
    pub x: f32,
    pub y: f32,
    pub a: f32,
    pub b: f32,
    pub r: f32,
    pub i: f32,
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
pub struct CommonTangentsIntermediateData {
    f_0: f32,
    g_0: f32,
    h_0: f32,
    f_1: f32,
    g_1: f32,
    h_1: f32,
    j: f32,
    w: f32,
    l: f32,
    o: f32,
    p: f32,
    v: f32,
    u: f32,
    m: f32,
}

impl Ellipse {
    pub fn center(&self) -> Point<f32> {
        (self.x, self.y).into()
    }

    pub fn rotation(&self) -> Complex<f32> {
        (self.r, self.i).into()
    }

    /// Into world space
    fn into_ws(&self, p: Point<f32>) -> Point<f32> {
        (p.rotated(Point::origin(), self.rotation())).absolute(self.center())
    }

    /// From world space
    fn from_ws(&self, p: Point<f32>) -> Point<f32> {
        p.relative(self.center())
            .rotated(Point::origin(), !self.rotation())
    }

    pub fn new(x: f32, y: f32, a: f32, b: f32, theta: f32) -> Self {
        Self {
            x,
            y,
            a,
            b,
            r: theta.cos(),
            i: theta.sin(),
        }
    }

    /// Focal point 0. Returns x, y of the point
    pub fn f0_old(&self) -> Point<f32> {
        if self.a.abs() > self.b.abs() {
            let major_axis = absmax(self.a, self.b);
            let minor_axis = absmin(self.a, self.b);
            let focal_len = (major_axis.pow(2.) - minor_axis.pow(2.)).sqrt();

            self.into_ws((0., focal_len).into())
        } else {
            let major_axis = absmax(self.a, self.b);
            let minor_axis = absmin(self.a, self.b);
            let focal_len = (major_axis.pow(2.) - minor_axis.pow(2.)).sqrt();

            self.into_ws((focal_len, 0.).into())
        }
    }

    pub fn f0(&self) -> Point<f32> {
        if self.a.abs() > self.b.abs() {
            let focal_len = (self.a.pow(2.) - self.b.pow(2.)).sqrt();
            self.into_ws((0., focal_len).into())
        } else {
            let focal_len = (self.b.pow(2.) - self.a.pow(2.)).sqrt();
            self.into_ws((focal_len, 0.).into())
        }
    }

    /// Focal point 1. Returns x, y of the point
    pub fn f1(&self) -> Point<f32> {
        if self.a.abs() > self.b.abs() {
            let major_axis = absmax(self.a, self.b);
            let minor_axis = absmin(self.a, self.b);
            let focal_len = (major_axis.pow(2.) - minor_axis.pow(2.)).sqrt();

            self.into_ws((0., -focal_len).into())
        } else {
            let major_axis = absmax(self.a, self.b);
            let minor_axis = absmin(self.a, self.b);
            let focal_len = (major_axis.pow(2.) - minor_axis.pow(2.)).sqrt();

            self.into_ws((-focal_len, 0.).into())
        }
    }

    pub fn perimeter(&self) -> f32 {
        let h = (self.a - self.b).sq() / (self.a + self.b).sq();
        PI * (self.a + self.b) * (1. + 3. * h / (10. + (4. - 3. * h).sqrt()))
    }

    pub fn from_foci(f0: Point<f32>, f1: Point<f32>, point_on_ellipse: Point<f32>) -> Ellipse {
        let c = (f0 - f1).len().into_inner() / 2.;

        // (sum / 2.)^2 = c^2 + b^2;
        // sum = (a-c)*2 + c*2
        // sum = a*2
        // a^2 == c^2 + b^2;

        let sum =
            (f0 - point_on_ellipse).len().into_inner() + (f1 - point_on_ellipse).len().into_inner();
        let a = sum / 2.;
        let b = (a.sq() - c.sq()).sqrt();

        let center = lerp(f0, f1, 0.5);

        let rot: Complex<f32> = (f0 - center).rotor();

        let rot: Complex<f32> = rot * Complex::from_polar(1., Angle::from_radians(-PI / 2.));

        Ellipse {
            x: *center.x(),
            y: *center.y(),
            a,
            b,
            r: *rot.real(),
            i: *rot.imag(),
        }
    }

    pub fn radius(&self, anomaly: Angle<f32>) -> f32 {
        let a = self.a;
        let b = self.b;
        a * b / f32::sqrt((b * anomaly.cos()).pow(2.) + (a * anomaly.sin()).pow(2.))
    }

    pub fn apoapsis(&self) -> f32 {
        if self.a.abs() > self.b.abs() {
            let focal_len = (self.a.pow(2.) - self.b.pow(2.)).sqrt();
            self.a.abs() + focal_len
        } else {
            let focal_len = (self.b.pow(2.) - self.a.pow(2.)).sqrt();
            self.b.abs() + focal_len
        }
    }

    pub fn periapsis(&self) -> f32 {
        if self.a.abs() > self.b.abs() {
            let focal_len = (self.a.pow(2.) - self.b.pow(2.)).sqrt();
            self.a.abs() - focal_len
        } else {
            let focal_len = (self.b.pow(2.) - self.a.pow(2.)).sqrt();
            self.b.abs() - focal_len
        }
    }

    pub fn point_on_ellipse(&self, anomaly: Angle<f32>) -> Point<f32> {
        self.into_ws((self.b.abs() * anomaly.sin(), self.a.abs() * anomaly.cos()).into())
    }

    pub fn acc(
        &self,
        anomaly: Angle<f32>,
        central_body_mass: Kg<f32>,
        gravitational_constant: f32,
    ) -> Vector<f32> {
        let vec = self.f0() - self.point_on_ellipse(anomaly);
        let vec_len = vec.len().into_inner();
        vec * central_body_mass.0 * gravitational_constant / vec_len.pow(3.)
    }

    pub fn tangential_velocity(
        &self,
        anomaly: Angle<f32>,
        central_body_mass: Kg<f32>,
        gravitational_constant: f32,
    ) -> Vector<f32> {
        // let p = self.point_on_ellipse(t);
        // let center = (self.x, self.y);

        let vec = self.f0() - self.point_on_ellipse(anomaly);
        let vec_len = vec.len().into_inner();

        // let radius = self.radius(t);
        // println!("radius: {}", radius);
        let velocity_module = f32::sqrt(
            gravitational_constant * central_body_mass.0 * (2. / vec_len - 1. / self.a.abs()),
        );

        // let tangent = (
        //     radius*f32::sin(t * 2. * PI),
        //     radius*f32::cos(t * 2. * PI),
        // );

        // let tangent = vmath::complex_mul( tangent, (self.i, self.r));

        let vel = Vector::from((
            // radius * f32::sin(t * 2. * PI + PI/2.),
            // radius * f32::cos(t * 2. * PI + PI/2.),
            self.b.abs() * anomaly.cos(),
            self.a.abs() * -anomaly.sin(),
        )) * self.rotation();

        vel.norm() * velocity_module
    }

    pub fn angular_velocity(
        &self,
        anomaly: Angle<f32>,
        central_body_mass: Kg<f32>,
        gravitational_constant: f32,
    ) -> DeltaAngle<f32> {
        let p = self.point_on_ellipse(anomaly);
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
        anomaly: Angle<f32>,
        central_body_mass: Kg<f32>,
        gravitational_constant: f32,
        vel: Vector<f32>,
    ) -> (Vector<f32>, Point<f32>) {
        let p = self.point_on_ellipse(anomaly);
        let f0 = self.f0();

        let _r = p - f0;
        let _v = vel;
        let _mu = central_body_mass.0 * gravitational_constant;
        let _r_len = _r.len().into_inner();
        let _v_len = _v.len().into_inner();
        let _h = _r.cross(_v);
        let _energy = _v_len.sq() / 2. - _mu / _r_len;
        let _a = -_mu / (2. * _energy);
        let _e = (_r * (_v_len.sq() - _mu / _r_len) - _v * _r.dot(_v)) * (1. / _mu);
        let _f1 = _e * -2. * _a;

        (_e, f0 + _f1)
    }

    /// Change ellipse in the way that f0, and position in `t` stays the same and velocity in `t` changeds to `vel`
    pub fn set_tangential_velicity(
        &self,
        anomaly: Angle<f32>,
        central_body_mass: Kg<f32>,
        gravitational_constant: f32,
        vel: Vector<f32>,
    ) -> (f32, f32, f32) {
        // consts
        let p = self.point_on_ellipse(anomaly);
        let f0 = self.f0();
        let vec_to_focus = f0 - p;
        #[allow(unused)]
        let vec_to_focus_len = vec_to_focus.len().into_inner();
        #[allow(unused)]
        let th = anomaly.radians();
        #[allow(unused)]
        let velocity_module = vel.len().into_inner();
        #[allow(unused)]
        let q = 1.
            / (4. / vec_to_focus_len
                - velocity_module.pow(2.) / gravitational_constant / central_body_mass.0);
        // ------

        // #[allow(unused)]
        // let (new_a, new_b, new_center, new_rotation): (f32, f32, Point<f32>, Complex<f32>) = todo!();

        /*

        #[allow(unreachable_code)]
        if new_a.abs() > new_b.abs() {
            assert!(
                new_center
                    == f0 - Vector::from((0., (q.pow(2.) - new_b.pow(2.)).sqrt())) * new_rotation
            );

            assert!(
                new_rotation
                    == Complex::div(
                        p - f0,
                        Vector::from((
                            new_b.abs() * f32::sin(th),
                            q.abs() * f32::cos(th) - (q.pow(2.) - new_b.pow(2.)).sqrt(),
                        ))
                    )
            ); // 3

            /*

                    {(a * c + b * d) / (c^2 + d^2) ,
                    (b * c - a * d) / (c^2 + d^2) }


                {
                    (new_b.abs() * f32::cos(th) * new_b.abs() * f32::sin(th) + q.abs() * -f32::sin(th) * (q.abs() * f32::cos(th) - (q.pow(2.) - new_b.pow(2.)).sqrt())) / ((new_b.abs() * f32::sin(th))^2 + (q.abs() * f32::cos(th) - (q.pow(2.) - new_b.pow(2.)).sqrt())^2) ,
                    (q.abs() * -f32::sin(th) * new_b.abs() * f32::sin(th) - new_b.abs() * f32::cos(th) * (q.abs() * f32::cos(th) - (q.pow(2.) - new_b.pow(2.)).sqrt())) / ((new_b.abs() * f32::sin(th))^2 + (q.abs() * f32::cos(th) - (q.pow(2.) - new_b.pow(2.)).sqrt())^2)
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
                         new_b.abs() * f32::cos(th),
                         q.abs() * -f32::sin(th)
                      )

                 / com(
                         new_b.abs() * f32::sin(th),
                         q.abs() * f32::cos(th) - (q.pow(2.) - new_b.pow(2.)).sqrt()
                      )

                == com(vel) / com(p - f0)

            );


        } else {
            assert!(self.into_ws(((new_b.pow(2.) - q.pow(2.)).sqrt(), 0.).into()) == f0);

            assert!(
                self.into_ws((new_b.abs() * f32::sin(th), q.abs() * f32::cos(th),).into()) == p
            );

            assert!(
                (Vector::from((new_b.abs() * f32::cos(th), q.abs() * -f32::sin(th),))
                    * new_rotation)
                    .norm()
                    == vel.norm()
            );
        }

        */

        let sin = |x| f32::sin(x);
        let cos = |x| f32::cos(x);
        let sqrt = |x| f32::sqrt(x);

        let hx = (vel.x() * (p.x() - f0.x()) + vel.y() * (p.y() - f0.y()))
            / ((p.x() - f0.x()).pow(2.) + (p.y() - f0.y()).pow(2.));
        let jx = hx * sin(th).pow(2.) - cos(th) * sin(th) - hx;
        let lx = hx * 2. * q.pow(2.) * cos(th).pow(2.) + q.pow(2.) * cos(th) * sin(th);
        let ox = (q.abs() * sin(th) + hx * 2. * q.abs() * cos(th)).pow(2.);

        // let xxxx =
        //                jx^2 * new_b^4
        //              + (ox + 2 * jx * lx) * new_b^2
        //              - q^2 * ox - lx^2
        //              == 0

        let desc = (ox + 2. * jx * lx).sq() + 4. * jx.sq() * (q.sq() * ox + lx.sq());

        let new_b0 = (-(ox + 2. * jx * lx) + sqrt(desc)) / (2. * jx.sq());
        let new_b1 = (-(ox + 2. * jx * lx) - sqrt(desc)) / (2. * jx.sq());

        println!(
            "ab: {}, {} -> {}, [{}, {}]",
            self.a, self.b, q, new_b0, new_b1
        );

        (q, new_b0, new_b1)
        // todo!()
    }

    pub fn accelerated(
        &self,
        anomaly: Angle<f32>,
        central_body_mass: Kg<f32>,
        gravitational_constant: f32,
        dt: Duration,
        acc: Vector<f32>,
    ) -> Ellipse {
        let f0 = self.f0();
        let p = self.point_on_ellipse(anomaly);
        let vel =
            self.tangential_velocity(anomaly, central_body_mass.clone(), gravitational_constant);
        let (_, new_f1) = self.f1_from_tangential_velocity(
            anomaly,
            central_body_mass,
            gravitational_constant,
            vel + acc * dt.as_secs_f32(),
        );
        Ellipse::from_foci(f0, new_f1, p)
    }

    pub fn eccentricity(&self) -> f32 {
        (1.0 - (self.b / self.a).powi(2)).sqrt()
    }

    pub fn time_between_anomalies(
        &self,
        anomaly0: Angle<f32>,
        anomaly1: Angle<f32>,
        central_body_mass: Kg<f32>,
        gravitational_constant: f32,
    ) -> Duration {
        time_between_true_anomalies(
            self.a,
            self.eccentricity(),
            anomaly0,
            anomaly1,
            central_body_mass.0 * gravitational_constant,
        )
    }

    /// Unlike `time_between_anomalies` can return negative time
    pub fn relative_time_between_anomalies(
        &self,
        anomaly0: Angle<f32>,
        anomaly1: Angle<f32>,
        central_body_mass: Kg<f32>,
        gravitational_constant: f32,
    ) -> RelativeDuration {
        relative_time_between_true_anomalies(
            self.a,
            self.eccentricity(),
            anomaly0,
            anomaly1,
            central_body_mass.0 * gravitational_constant,
        )
    }

    pub(crate) fn eq_no_rot(&self, x: f32, y: f32) -> f32 {
        (x - self.x).pow(2.) / self.a.pow(2.) + (y - self.y).pow(2.) / self.b.pow(2.) - 1.
    }

    pub fn eq(&self) -> impl FnOnce(f32, f32) -> f32 {
        let x_0 = self.x;
        let y_0 = self.y;
        let a = self.a;
        let b = self.b;
        let r = self.r;
        let i = self.i;
        move |x, y| {
            (r * (x - x_0) - i * (y - y_0)).pow(2.) / a.pow(2.)
                + (r * (y - y_0) + i * (x - x_0)).pow(2.) / b.pow(2.)
                - 1.
        }
    }

    /// (r*(x - x_0) - i*(y - y_0))^2 / a^2 + (r*(y - y_0) + i*(x - x_0))^2 / b^2 - 1
    /// y = k*x+d
    ///
    /// (r * (x - x_0) - i * (k * x + d - y_0))^2 / a^2 + (r * (k * x + d - y_0) + i * (x - x_0))^2 / b^2 - 1
    /// (r * (x - x_0) - e * (k * x + d - y_0))^2 / a^2 + (r * (k * x + d - y_0) + e * (x - x_0))^2 / b^2 - 1
    pub(crate) fn intersection_line_eq(&self, line: Line) -> impl FnOnce(f32) -> f32 {
        let x_0 = self.x;
        let y_0 = self.y;
        let a = self.a;
        let b = self.b;
        let r = self.r;
        let i = self.i;
        let k = line.k;
        let d = line.d;
        move |x| {
            (r * (x - x_0) - i * (k * x + d - y_0)).pow(2.) / a.pow(2.)
                + (r * (k * x + d - y_0) + i * (x - x_0)).pow(2.) / b.pow(2.)
                - 1.
        }
    }

    /// returns discriminant of intersection equesion with `line`
    /// -(4 * (a^2 * (-k^2 * r^2 - 2 * i * k * r - i^2) + b^2 * (-i^2 * k^2 + 2 * i * k * r - r^2) + (r^4 + 2 * i^2 * r^2 + i^4) * (d^2 + 2 * d * (k * x_0 - y_0) + k^2 * x_0^2 - 2 * k * x_0 * y_0 + y_0^2))) / (a^2 * b^2)
    /// -(4 * (a_0^2 * (-k^2 * r_0^2 - 2 * i_0 * k * r_0 - i_0^2) + b_0^2 * (-i_0^2 * k^2 + 2 * i_0 * k * r_0 - r_0^2) + (r_0^4 + 2 * i_0^2 * r_0^2 + i_0^4) * (d^2 + 2 * d * (k * x_0 - y_0) + k^2 * x_0^2 - 2 * k * x_0 * y_0 + y_0^2))) / (a_0^2 * b_0^2)
    /// -(4 * (a_1^2 * (-k^2 * r_1^2 - 2 * i_1 * k * r_1 - i_1^2) + b_1^2 * (-i_1^2 * k^2 + 2 * i_1 * k * r_1 - r_1^2) + (r_1^4 + 2 * i_1^2 * r_1^2 + i_1^4) * (d^2 + 2 * d * (k * x_1 - y_1) + k^2 * x_1^2 - 2 * k * x_1 * y_1 + y_1^2))) / (a_1^2 * b_1^2)
    pub fn intersection_discriminant(&self, line: Line) -> f32 {
        let x_0 = self.x;
        let y_0 = self.y;
        let a = self.a;
        let b = self.b;
        let r = self.r;
        let i = self.i;
        let k = line.k;
        let d = line.d;

        4. * ((a * (r * k + i)).pow(2.) + (b * (i * k - r)).pow(2.) - (d + k * x_0).pow(2.)
            + 2. * d * y_0
            + 2. * k * x_0 * y_0
            - y_0.pow(2.))
            / (a * b).pow(2.)
    }

    /// returns `d` by given `k` where `y = kx + d` is a tangent to ellipse
    pub fn tangent_d(&self, k: f32) -> (f32, f32) {
        let x_0 = self.x;
        let y_0 = self.y;
        let a = self.a;
        let b = self.b;
        let r = self.r;
        let i = self.i;

        let discriminant = (a * (r * k + i)).pow(2.) + (b * (i * k - r)).pow(2.);
        let base = -k * x_0 + y_0;

        (base + discriminant.sqrt(), base - discriminant.sqrt())
    }

    /// intersection of this function with y = 0 is where common outer tangents are
    pub fn outer_tangents_fun(&self, rhs: &Ellipse, k: f32) -> (f32, f32) {
        let x_0 = self.x;
        let y_0 = self.y;
        let a_0 = self.a;
        let b_0 = self.b;
        let r_0 = self.r;
        let i_0 = self.i;

        let x_1 = rhs.x;
        let y_1 = rhs.y;
        let a_1 = rhs.a;
        let b_1 = rhs.b;
        let r_1 = rhs.r;
        let i_1 = rhs.i;

        let discriminant_0 = (a_0 * (r_0 * k + i_0)).pow(2.) + (b_0 * (i_0 * k - r_0)).pow(2.);
        let discriminant_1 = (a_1 * (r_1 * k + i_1)).pow(2.) + (b_1 * (i_1 * k - r_1)).pow(2.);

        let lhs = k * (x_1 - x_0) + y_0 - y_1;
        let rhs = discriminant_1.sqrt() - discriminant_0.sqrt();

        (lhs - rhs, lhs + rhs)
    }

    pub fn common_tangents_intermediate_data(
        &self,
        rhs: &Ellipse,
    ) -> CommonTangentsIntermediateData {
        let x_0 = self.x;
        let y_0 = self.y;
        let a_0 = self.a;
        let b_0 = self.b;
        let r_0 = self.r;
        let i_0 = self.i;

        let x_1 = rhs.x;
        let y_1 = rhs.y;
        let a_1 = rhs.a;
        let b_1 = rhs.b;
        let r_1 = rhs.r;
        let i_1 = rhs.i;

        let c: f32 = 1.;

        let f_0 = ((a_0 * r_0).pow(2.) + (b_0 * i_0).pow(2.)) / c;
        let g_0 = (2. * i_0 * r_0 * (a_0.pow(2.) - b_0.pow(2.))) / c;
        let h_0 = ((a_0 * i_0).pow(2.) + (b_0 * r_0).pow(2.)) / c;

        let f_1 = ((a_1 * r_1).pow(2.) + (b_1 * i_1).pow(2.)) / c;
        let g_1 = (2. * i_1 * r_1 * (a_1.pow(2.) - b_1.pow(2.))) / c;
        let h_1 = ((a_1 * i_1).pow(2.) + (b_1 * r_1).pow(2.)) / c;

        let dx = (x_1 - x_0).ssq() / c;
        let dy = (y_1 - y_0).ssq() / c;

        // println!(
        //     "before: {}, {}, {}, {}, {}, {}, {}, {}",
        //     f_0, g_0, h_0, f_1, g_1, h_1, dx, dy
        // );
        let [f_0, g_0, h_0, f_1, g_1, h_1, dx, dy] =
            notmalize_array_around_one([f_0, g_0, h_0, f_1, g_1, h_1, dx, dy]);
        // println!(
        //     "after : {}, {}, {}, {}, {}, {}, {}, {}",
        //     f_0, g_0, h_0, f_1, g_1, h_1, dx, dy
        // );

        let dx = dx.ssqrt();
        let dy = dy.ssqrt();

        let j = f_1 + f_0 - dx.pow(2.);
        let w = g_1 + g_0 + 2. * dx * dy;
        let l = h_1 + h_0 - dy.pow(2.);

        let o = j.pow(2.) - 4. * f_1 * f_0;
        let p = 2. * j * w - 4. * f_1 * g_0 - 4. * f_0 * g_1;
        let v = w.pow(2.) + 2. * j * l - 4. * f_1 * h_0 - 4. * g_1 * g_0 - 4. * h_1 * f_0;
        let u = 2. * w * l - 4. * g_1 * h_0 - 4. * h_1 * g_0;
        let m = l.pow(2.) - 4. * h_1 * h_0;

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

    pub fn common_tangents(&self, rhs: &Ellipse) -> Vec<(Line, TangentDirection)> {
        let id = self.common_tangents_intermediate_data(rhs);

        fn pp<'a>(
            e0: &'a Ellipse,
            e1: &'a Ellipse,
            j: f32,
            w: f32,
            l: f32,
            roots: &'a [f32],
        ) -> impl Iterator<Item = (Line, TangentDirection)> + 'a {
            //println!("roots: {:?}", roots);
            roots
                .iter()
                .filter(move |k| k.pow(2.) * j + *k * w + l >= 0.)
                .map(|k| {
                    let d_0 = e0.tangent_d(*k);
                    let d_1 = e1.tangent_d(*k);

                    let err = 0.1;

                    let mut vec = Vec::new();

                    if (d_0.0 - d_1.0).abs() < err || (d_0.0 - d_1.1).abs() < err {
                        vec.push((Line { k: *k, d: d_0.0 }, TangentDirection::Left));
                    }
                    if (d_0.1 - d_1.0).abs() < err || (d_0.1 - d_1.1).abs() < err {
                        vec.push((Line { k: *k, d: d_0.1 }, TangentDirection::Right));
                    }
                    vec
                })
                .flatten()
        }

        //println!("pol: {}, {}, {}, {}, {}", o, p, v, u, m);
        //let norm = notmalize_array([id.o, id.p, id.v, id.u, id.m]);
        //println!("norm: {:?}", norm);

        let poly = Polynomial::<f64>::new(vec![
            id.o as f64,
            id.p as f64,
            id.v as f64,
            id.u as f64,
            id.m as f64,
        ]);

        //println!("roots2: {:?}", );

        let res = match poly.roots() {
            rustnomial::Roots::NoRoots => pp(self, rhs, id.j, id.w, id.l, &[]).collect(),
            rustnomial::Roots::NoRootsFound => pp(self, rhs, id.j, id.w, id.l, &[]).collect(),
            rustnomial::Roots::OneRealRoot(root) => {
                pp(self, rhs, id.j, id.w, id.l, &[root as f32]).collect()
            }
            rustnomial::Roots::TwoRealRoots(r0, r1) => {
                pp(self, rhs, id.j, id.w, id.l, &[r0 as f32, r1 as f32]).collect()
            }
            rustnomial::Roots::ThreeRealRoots(r0, r1, r2) => pp(
                self,
                rhs,
                id.j,
                id.w,
                id.l,
                &[r0 as f32, r1 as f32, r2 as f32],
            )
            .collect(),
            rustnomial::Roots::ManyRealRoots(roots) => pp(
                self,
                rhs,
                id.j,
                id.w,
                id.l,
                &roots.iter().map(|x| *x as f32).collect::<Vec<_>>(),
            )
            .collect(),
            rustnomial::Roots::OneComplexRoot(_) => pp(self, rhs, id.j, id.w, id.l, &[]).collect(),
            rustnomial::Roots::TwoComplexRoots(_, _) => {
                pp(self, rhs, id.j, id.w, id.l, &[]).collect()
            }
            rustnomial::Roots::ThreeComplexRoots(_, _, _) => {
                pp(self, rhs, id.j, id.w, id.l, &[]).collect()
            }
            rustnomial::Roots::ManyComplexRoots(_) => {
                pp(self, rhs, id.j, id.w, id.l, &[]).collect()
            }
            rustnomial::Roots::InfiniteRoots => pp(self, rhs, id.j, id.w, id.l, &[]).collect(),
            rustnomial::Roots::OnlyRealRoots(roots) => pp(
                self,
                rhs,
                id.j,
                id.w,
                id.l,
                &roots.iter().map(|x| *x as f32).collect::<Vec<_>>(),
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

        //     let final_val = k.pow(4.) * o + k.pow(3.) * p + k.pow(2.) * v + k * u + m;

        //     eq(final_val, 0.)
        // } else {
        //     f32::MAX
        // };

        //(eq0 / 100., eq1 / 20000000.)
    }

    pub fn tangent_k_alg(&self, rhs: &Ellipse, k: f32) -> (f32, f32) {
        let x_0 = self.x;
        let y_0 = self.y;
        let a_0 = self.a;
        let b_0 = self.b;
        let r_0 = self.r;
        let i_0 = self.i;

        let x_1 = rhs.x;
        let y_1 = rhs.y;
        let a_1 = rhs.a;
        let b_1 = rhs.b;
        let r_1 = rhs.r;
        let i_1 = rhs.i;

        let eq = |left: f32, right: f32| (left - right).abs();

        let f_0 = (a_0 * r_0).pow(2.) + (b_0 * i_0).pow(2.);
        let g_0 = 2. * i_0 * r_0 * (a_0.pow(2.) - b_0.pow(2.));
        let h_0 = (a_0 * i_0).pow(2.) + (b_0 * r_0).pow(2.);

        let _discriminant_0 = k.pow(2.) * f_0 + k * g_0 + h_0;

        //let discriminant_0
        //    = (a_0 * (r_0 * k + i_0)).pow(2.)
        //    + (b_0 * (i_0 * k - r_0)).pow(2.);

        // (a^2 + b^2).sqrt() - (c^2 + d^2).sqrt() = z;
        // (a^2 + b^2) + (c^2 + d^2) - (a^2 + b^2).sqrt() * (c^2 + d^2).sqrt() = z^2
        // (a^2 + b^2 + c^2 + d^2 - z^2)^2 = (a^2 + b^2)(c^2 + d^2)
        //

        let f_1 = (a_1 * r_1).pow(2.) + (b_1 * i_1).pow(2.);
        let g_1 = 2. * i_1 * r_1 * (a_1.pow(2.) - b_1.pow(2.));
        let h_1 = (a_1 * i_1).pow(2.) + (b_1 * r_1).pow(2.);

        let _discriminant_1 = k.pow(2.) * f_1 + k * g_1 + h_1;

        //let discriminant_1
        //    = (a_1 * (r_1 * k + i_1)).pow(2.)
        //    + (b_1 * (i_1 * k - r_1)).pow(2.);

        let rhs =
            (k.pow(2.) * f_1 + k * g_1 + h_1).sqrt() - (k.pow(2.) * f_0 + k * g_0 + h_0).sqrt();

        let _discriminant_0 = (a_0 * (r_0 * k + i_0)).pow(2.) + (b_0 * (i_0 * k - r_0)).pow(2.);
        let _discriminant_1 = (a_1 * (r_1 * k + i_1)).pow(2.) + (b_1 * (i_1 * k - r_1)).pow(2.);

        //= a.pow(2.)
        //+ b.pow(2.)
        //- 2. * a * b
        //- 2. * a * c.pow(2.)
        //- 2. * b * c.pow(2.)
        //+ c.pow(4.)

        let eq0 = eq(
            (k * (x_1 - x_0) + y_0 - y_1).pow(2.),
            ((k.pow(2.) * f_1 + k * g_1 + h_1).sqrt() - (k.pow(2.) * f_0 + k * g_0 + h_0).sqrt())
                .pow(2.),
        );

        let f_0 = (a_0 * r_0).pow(2.) + (b_0 * i_0).pow(2.);
        let g_0 = 2. * i_0 * r_0 * (a_0.pow(2.) - b_0.pow(2.));
        let h_0 = (a_0 * i_0).pow(2.) + (b_0 * r_0).pow(2.);

        let f_1 = (a_1 * r_1).pow(2.) + (b_1 * i_1).pow(2.);
        let g_1 = 2. * i_1 * r_1 * (a_1.pow(2.) - b_1.pow(2.));
        let h_1 = (a_1 * i_1).pow(2.) + (b_1 * r_1).pow(2.);

        let j = f_1 + f_0 - (x_0 - x_1).pow(2.);
        let w = g_1 + g_0 - 2. * (x_0 - x_1) * (y_1 - y_0);
        let l = h_1 + h_0 - (y_1 - y_0).pow(2.);

        let eq1 = if k.pow(2.) * j + k * w + l >= 0. {
            let o = j.pow(2.) - 4. * f_1 * f_0;
            let p = 2. * j * w - 4. * f_1 * g_0 - 4. * f_0 * g_1;
            let v = 2. * j * l + w.pow(2.) - 4. * f_1 * h_0 - 4. * g_1 * g_0 - 4. * h_1 * f_0;
            let u = 2. * w * l - 4. * g_1 * h_0 - 4. * h_1 * g_0;
            let m = l.pow(2.) - 4. * h_1 * h_0;

            let final_val = k.pow(4.) * o + k.pow(3.) * p + k.pow(2.) * v + k * u + m;

            eq(final_val, 0.)
        } else {
            f32::MAX
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

        (eq0 / 100., eq1 / 20000000.)
    }
}

fn time_between_true_anomalies(
    major_axis: f32,
    eccentricity: f32,
    nu1: Angle<f32>,
    nu2: Angle<f32>,
    mu: f32,
) -> Duration {
    let e1 = true_anomaly_to_eccentric(eccentricity, nu1);
    let e2 = true_anomaly_to_eccentric(eccentricity, nu2);

    let m1 = eccentric_anomaly_to_mean(eccentricity, e1);
    let m2 = eccentric_anomaly_to_mean(eccentricity, e2);

    // Mean motion (n = sqrt(mu / a^3))
    let n = (mu / major_axis.powi(3)).sqrt();

    let mut delta_m = m2 - m1;

    // Ensure the time is positive if traveling forward
    if delta_m < 0.0 {
        delta_m += 2.0 * PI;
    }

    Duration::from_secs_f32(delta_m / n)
}

fn relative_time_between_true_anomalies(
    major_axis: f32,
    eccentricity: f32,
    nu1: Angle<f32>,
    nu2: Angle<f32>,
    mu: f32,
) -> RelativeDuration {
    let e1 = true_anomaly_to_eccentric(eccentricity, nu1);
    let e2 = true_anomaly_to_eccentric(eccentricity, nu2);

    let m1 = eccentric_anomaly_to_mean(eccentricity, e1);
    let m2 = eccentric_anomaly_to_mean(eccentricity, e2);

    // Mean motion (n = sqrt(mu / a^3))
    let n = (mu / major_axis.powi(3)).sqrt();

    RelativeDuration::from_secs_f32((m2 - m1) / n)
}

fn true_anomaly_to_eccentric(eccentricity: f32, anomaly: Angle<f32>) -> f32 {
    let factor = ((1.0 - eccentricity) / (1.0 + eccentricity)).sqrt();
    2.0 * (factor * (anomaly.radians() / 2.0).tan()).atan()
}

fn eccentric_anomaly_to_mean(eccentricity: f32, e_anom: f32) -> f32 {
    e_anom - eccentricity * e_anom.sin()
}

#[cfg(test)]
mod tests {
    use super::Ellipse;
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

    static E0: Ellipse = Ellipse {
        x: 100.0,
        y: 100.0,
        a: 40.0,
        b: 70.0,
        r: 0.9659258,
        i: 0.25881904,
    };

    static E1: Ellipse = Ellipse {
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

    fn eq(left: f32, right: f32) -> f32 {
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
            //eq(q.pow(2.), a + 2. * a.sqrt() * b.sqrt() + b)

            //eq(q.pow(4.) + 2. * a * q.pow(2.) - 2. * b * q.pow(2.) + a.pow(2.) + b.pow(2.) - 6. * a * b, 0.)
        };

        // q = (a - b) / c

        // q = a.sqrt() + b.sqrt();
        // q.pow(2.) = a + 2 * a.sqrt() * b.sqrt() + b;
        // q.pow(2.) - a - b = 2 * a.sqrt() * b.sqrt()
        // (q.pow(2.) + (a - b)).pow(2.) = 4 * a * b
        // q.pow(4.) + 2 * c.pow(2.) * (a - b) + (a - b)^2 = 4 * a * b
        // c.pow(4.) + 2 * a * c.pow(2.) - 2 * b * c.pow(2.) + a.pow(2.) - 2*a*b + b.pow(2.) - 4*a*b = 0
        // c.pow(4.) + 2 * a * c.pow(2.) - 2 * b * c.pow(2.) + a.pow(2.) + b.pow(2.) - 6*a*b = 0

        let fun3 = |a: f32, b: f32, c: f32| {
            eq(
                a.pow(2.) + b.pow(2.) - 2. * a * b - 2. * a * c.pow(2.) - 2. * b * c.pow(2.)
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
        let do_assert = |fun: fn(f32, f32, f32) -> f32| {
            assert_eq_err!(fun(9., 4., 1.), 0., 0.001);
            assert_eq_err!(fun(16., 4., 2.), 0., 0.001);
            assert_eq_err!(fun(16., 9., 1.), 0., 0.001);
            assert_eq_err!(fun(9., 16., -1.), 0., 0.001);
            assert_ne_err!(fun(9., 16., 1.), 0., 0.001);
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
    //         530586800_f32 / den,
    //         -3043730000_f32 / den,
    //         5108047400_f32 / den,
    //         -3046621000_f32 / den,
    //         502631040_f32 / den,
    //     );
    //     assert_eq!(roots, Roots::Four([0.26496, 0.81798, 1.30545, 3.34813]))
    // }
}
