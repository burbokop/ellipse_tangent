use std::mem::Discriminant;

use nannou::prelude::Pow;

use crate::line::Line;

#[derive(Debug, Clone)]
pub(crate) struct Ellipse {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) a: f32,
    pub(crate) b: f32,
    pub(crate) r: f32,
    pub(crate) i: f32,
}

// (x-x0)^2 / a^2 + (y-y0)^2 / b^2 = 1

//x+yI = (r+iI)*(ux+uyI) = r*ux + r*uy*I + i*I*ux - i*uy = (r*ux - i*uy) + (r*uy + i*ux)*I
//x+yI = (ux+uyI)*(r+iI) = ux*r + i*ux*I + r*I*uy - i*uy = (r*ux - i*uy) + (i*uy + r*ux)*I

// (x-x0)^2 / a^2 + (y-y0)^2 / b^2 = 1

// (r*x - i*y - x0)^2 / a^2 + (r*y + i*x - y0)^2 / b^2 = 1

// (r*(x - x0) - i*(y - y0))^2 / a^2 + (r*(y - y0) + i*(x - x0))^2 / b^2 = 1

impl Ellipse {
    pub(crate) fn new(x: f32, y: f32, a: f32, b: f32, theta: f32) -> Self {
        Self {
            x,
            y,
            a,
            b,
            r: theta.cos(),
            i: theta.sin(),
        }
    }

    pub(crate) fn eq_no_rot(&self, x: f32, y: f32) -> f32 {
        (x - self.x).pow(2.) / self.a.pow(2.) + (y - self.y).pow(2.) / self.b.pow(2.) - 1.
    }

    pub(crate) fn eq(&self) -> impl FnOnce(f32, f32) -> f32 {
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
    pub(crate) fn intersection_discriminant(&self, line: Line) -> f32 {
        let x_0 = self.x;
        let y_0 = self.y;
        let a = self.a;
        let b = self.b;
        let r = self.r;
        let i = self.i;
        let k = line.k;
        let d = line.d;

        -(4. * (a.pow(2.) * (-k.pow(2.) * r.pow(2.) - 2. * i * k * r - i.pow(2.))
            + b.pow(2.) * (-i.pow(2.) * k.pow(2.) + 2. * i * k * r - r.pow(2.))
            + (r.pow(4.) + 2. * i.pow(2.) * r.pow(2.) + i.pow(4.))
                * (d.pow(2.) + 2. * d * (k * x_0 - y_0) + k.pow(2.) * x_0.pow(2.)
                    - 2. * k * x_0 * y_0
                    + y_0.pow(2.))))
            / (a.pow(2.) * b.pow(2.))
    }

    /// returns `d` by given `k` where `y = kx + d` is a tangent to ellipse
    pub(crate) fn tangent_d(&self, k: f32) -> (f32, f32) {
        let x_0 = self.x;
        let y_0 = self.y;
        let a = self.a;
        let b = self.b;
        let r = self.r;
        let i = self.i;
        let q = b / (r.pow(2.) + i.pow(2.));

        let discriminant = (a * (r * k + i)).pow(2.) + (q * (i * k - r)).pow(2.);
        let base = -k * x_0 + y_0;

        (base + discriminant.sqrt(), base - discriminant.sqrt())
    }

    /// intersection of this function with y = 0 is where common outer tangents are
    pub(crate) fn outer_tangents_fun(&self, rhs: &Ellipse, k: f32) -> (f32, f32) {
        let x_0 = self.x;
        let y_0 = self.y;
        let a_0 = self.a;
        let b_0 = self.b;
        let r_0 = self.r;
        let i_0 = self.i;
        let q_0 = b_0 / (r_0.pow(2.) + i_0.pow(2.));

        let x_1 = rhs.x;
        let y_1 = rhs.y;
        let a_1 = rhs.a;
        let b_1 = rhs.b;
        let r_1 = rhs.r;
        let i_1 = rhs.i;
        let q_1 = b_1 / (r_1.pow(2.) + i_1.pow(2.));

        let discriminant_0 = (a_0 * (r_0 * k + i_0)).pow(2.) + (q_0 * (i_0 * k - r_0)).pow(2.);
        let discriminant_1 = (a_1 * (r_1 * k + i_1)).pow(2.) + (q_1 * (i_1 * k - r_1)).pow(2.);

        let lhs = k * (x_1 - x_0) + y_0 - y_1;
        let rhs = discriminant_1.sqrt() - discriminant_0.sqrt();

        (lhs - rhs, lhs + rhs)
    }

    pub(crate) fn common_tangents(&self, rhs: &Ellipse) -> (Line, Line) {
        let x_0 = self.x;
        let y_0 = self.y;
        let a_0 = self.a;
        let b_0 = self.b;
        let r_0 = self.r;
        let i_0 = self.i;
        let q_0 = b_0 / (r_0.pow(2.) + i_0.pow(2.));

        let x_1 = rhs.x;
        let y_1 = rhs.y;
        let a_1 = rhs.a;
        let b_1 = rhs.b;
        let r_1 = rhs.r;
        let i_1 = rhs.i;
        let q_1 = b_1 / (r_1.pow(2.) + i_1.pow(2.));

        let mut k: f32 = 0.;

        let mut prev_err: (f32, f32) = (0., 0.);
        loop {
            let discriminant_0 = (a_0 * (r_0 * k + i_0)).pow(2.) + (q_0 * (i_0 * k - r_0)).pow(2.);
            let discriminant_1 = (a_1 * (r_1 * k + i_1)).pow(2.) + (q_1 * (i_1 * k - r_1)).pow(2.);

            let lhs = k * (x_1 - x_0) + y_0 - y_1;
            let rhs = discriminant_1.sqrt() - discriminant_0.sqrt();

            let err: (f32, f32) = (lhs - rhs, lhs + rhs);

            prev_err = err
        }

        (Line { k: 0., d: 0. }, Line { k: 0., d: 0. })
    }

    pub(crate) fn tangent_k_alg(&self, rhs: &Ellipse, k: f32) -> (f32, f32) {
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

        let q_0 = b_0 / (r_0.pow(2.) + i_0.pow(2.));
        let q_1 = b_1 / (r_1.pow(2.) + i_1.pow(2.));

        let eq = |left: f32, right: f32| (left - right).abs();

        let f_0 = (a_0 * r_0).pow(2.) + (q_0 * i_0).pow(2.);
        let g_0 = 2. * i_0 * r_0 * (a_0 - q_0) * (a_0 + q_0);
        let h_0 = (a_0 * i_0).pow(2.) + (q_0 * r_0).pow(2.);

        let discriminant_0 = k.pow(2.) * f_0 + k * g_0 + h_0;

        //let discriminant_0
        //    = (a_0 * (r_0 * k + i_0)).pow(2.)
        //    + (q_0 * (i_0 * k - r_0)).pow(2.);

        // (a^2 + b^2).sqrt() - (c^2 + d^2).sqrt() = z;
        // (a^2 + b^2) + (c^2 + d^2) - (a^2 + b^2).sqrt() * (c^2 + d^2).sqrt() = z^2
        // (a^2 + b^2 + c^2 + d^2 - z^2)^2 = (a^2 + b^2)(c^2 + d^2)
        //

        let f_1 = a_1.pow(2.) * r_1.pow(2.) + q_1.pow(2.) * i_1.pow(2.);
        let g_1 = 2. * i_1 * r_1 * (a_1.pow(2.) - q_1.pow(2.));
        let h_1 = a_1.pow(2.) * i_1.pow(2.) + q_1.pow(2.) * r_1.pow(2.);

        let discriminant_1 = k.pow(2.) * f_1 + k * g_1 + h_1;

        //let discriminant_1
        //    = (a_1 * (r_1 * k + i_1)).pow(2.)
        //    + (q_1 * (i_1 * k - r_1)).pow(2.);

        // a^2 + b^2 / c^2
        // ((ac)^2 + b^2) / c^2

        let d_d = 4. * (k * x_1 - y_1).pow(2.)
            - 4. * (k.pow(2.) * x_1.pow(2.) - 2. * k * x_1 * y_1
                + y_1.pow(2.)
                + (a_1.pow(2.) * (-k.pow(2.) * r_1.pow(2.) - 2. * i_1 * k * r_1 - i_1.pow(2.))
                    + b_1.pow(2.) * (-i_1.pow(2.) * k.pow(2.) + 2. * i_1 * k * r_1 - r_1.pow(2.)))
                    / (r_1.pow(4.) + 2. * i_1.pow(2.) * r_1.pow(2.) + i_1.pow(4.)))
            - 4. * (k * x_0 - y_0).pow(2.)
            + 4. * (k.pow(2.) * x_0.pow(2.) - 2. * k * x_0 * y_0
                + y_0.pow(2.)
                + (a_0.pow(2.) * (-k.pow(2.) * r_0.pow(2.) - 2. * i_0 * k * r_0 - i_0.pow(2.))
                    + b_0.pow(2.) * (-i_0.pow(2.) * k.pow(2.) + 2. * i_0 * k * r_0 - r_0.pow(2.)))
                    / (r_0.pow(4.) + 2. * i_0.pow(2.) * r_0.pow(2.) + i_0.pow(4.)));

        //c = a.sqrt() - b.sqrt();
        //c^2 = a + b - 2 * a.sqrt() * b.sqrt()
        //c^2 - a - b = - 2 * a.sqrt() * b.sqrt()
        //a + b - c^2 = 2 * a.sqrt() * b.sqrt()
        //(a + b)^2 - 2(a + b)c^2 + c^4 = 4ab
        //a^2 + b^2 + 2ab - 2ac^2 - 2bc^2 + c^4 = 4ab
        //a^2 + b^2 - 2ab - 2ac^2 - 2bc^2 + c^4 = 0

        //a = 1
        //b = 0

        let rhs =
            (k.pow(2.) * f_1 + k * g_1 + h_1).sqrt() - (k.pow(2.) * f_0 + k * g_0 + h_0).sqrt();

        let try_0 = (k.pow(2.) * f_1 + k * g_1 + h_1).pow(2.)
            + (k.pow(2.) * f_0 + k * g_0 + h_0).pow(2.)
            - 2. * (k.pow(2.) * f_1 + k * g_1 + h_1) * (k.pow(2.) * f_0 + k * g_0 + h_0)
            - 2. * (k.pow(2.) * f_1 + k * g_1 + h_1) * (k * (x_1 - x_0) + y_0 - y_1).pow(2.)
            - 2. * (k.pow(2.) * f_0 + k * g_0 + h_0) * (k * (x_1 - x_0) + y_0 - y_1).pow(2.)
            + (k * (x_1 - x_0) + y_0 - y_1).pow(4.);

        let try_1 = (k.pow(2.) * f_1 + k * g_1 + h_1).pow(2.)
            + (k.pow(2.) * f_0 + k * g_0 + h_0).pow(2.)
            - 2. * (k.pow(2.) * f_1 + k * g_1 + h_1) * (k.pow(2.) * f_0 + k * g_0 + h_0)
            - 2. * (k.pow(2.) * f_1 + k * g_1 + h_1) * (k * (x_0 - x_1) + y_1 - y_0).pow(2.)
            - 2. * (k.pow(2.) * f_0 + k * g_0 + h_0) * (k * (x_0 - x_1) + y_1 - y_0).pow(2.)
            + (k * (x_0 - x_1) + y_1 - y_0).pow(4.);

        // let try_0 = ((a_1 * (r_1 * k + i_1)).pow(2.) + (q_1 * (i_1 * k - r_1)).pow(2.)).pow(2.)
        //     + ((a_0 * (r_0 * k + i_0)).pow(2.) + (q_0 * (i_0 * k - r_0)).pow(2.)).pow(2.)
        //     - 2. * ((a_1 * (r_1 * k + i_1)).pow(2.) + (q_1 * (i_1 * k - r_1)).pow(2.)) * ((a_0 * (r_0 * k + i_0)).pow(2.) + (q_0 * (i_0 * k - r_0)).pow(2.))
        //     - 2. * ((a_1 * (r_1 * k + i_1)).pow(2.) + (q_1 * (i_1 * k - r_1)).pow(2.)) * (k * (x_1 - x_0) + y_0 - y_1).pow(2.)
        //     - 2. * ((a_0 * (r_0 * k + i_0)).pow(2.) + (q_0 * (i_0 * k - r_0)).pow(2.)) * (k * (x_1 - x_0) + y_0 - y_1).pow(2.)
        //     + (k * (x_1 - x_0) + y_0 - y_1).pow(4.);

        // let try_1 = ((a_1 * (r_1 * k + i_1)).pow(2.) + (q_1 * (i_1 * k - r_1)).pow(2.)).pow(2.)
        //     + ((a_0 * (r_0 * k + i_0)).pow(2.) + (q_0 * (i_0 * k - r_0)).pow(2.)).pow(2.)
        //     - 2. * ((a_1 * (r_1 * k + i_1)).pow(2.) + (q_1 * (i_1 * k - r_1)).pow(2.)) * ((a_0 * (r_0 * k + i_0)).pow(2.) + (q_0 * (i_0 * k - r_0)).pow(2.))
        //     - 2. * ((a_1 * (r_1 * k + i_1)).pow(2.) + (q_1 * (i_1 * k - r_1)).pow(2.)) * (k * (x_0 - x_1) + y_1 - y_0).pow(2.)
        //     - 2. * ((a_0 * (r_0 * k + i_0)).pow(2.) + (q_0 * (i_0 * k - r_0)).pow(2.)) * (k * (x_0 - x_1) + y_1 - y_0).pow(2.)
        //     + (k * (x_0 - x_1) + y_1 - y_0).pow(4.);

        let discriminant_0 = (a_0 * (r_0 * k + i_0)).pow(2.) + (q_0 * (i_0 * k - r_0)).pow(2.);
        let discriminant_1 = (a_1 * (r_1 * k + i_1)).pow(2.) + (q_1 * (i_1 * k - r_1)).pow(2.);

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

        (k * (x_0 - x_1) + y_1 - y_0).pow(2.);
        k.pow(2.) * (x_0 - x_1).pow(2.) + 2. * k * (x_0 - x_1) * (y_1 - y_0) + (y_1 - y_0).pow(2.);

        let j = f_1 + f_0 - (x_0 - x_1).pow(2.);
        let w = g_1 + g_0 - 2. * (x_0 - x_1) * (y_1 - y_0);
        let l = h_1 + h_0 - (y_1 - y_0).pow(2.);

        (k.pow(2.) * j + k * w + l).pow(2.);
        k.pow(4.) * j.pow(2.) + 2. * k.pow(2.) * j * (k * w + l) + (k * w + l).pow(2.);

        (k.pow(2.) * f_1 + k * g_1 + h_1) * (k.pow(2.) * f_0 + k * g_0 + h_0);

        //= k.pow(2.) * f_1 * k.pow(2.) * f_0
        //+ k.pow(2.) * f_1 * k * g_0
        //+ k.pow(2.) * f_1 * h_0
        //+ k * g_1 * k.pow(2.) * f_0
        //+ k * g_1 * k * g_0
        //+ k * g_1 * h_0
        //+ h_1 * k.pow(2.) * f_0
        //+ h_1 * k * g_0
        //+ h_1 * h_0

        let eq1 = if (k.pow(2.) * j + k * w + l) >= 0. {
            let o = j.pow(2.) - 4. * f_1 * f_0;
            let p = 2. * j * w - 4. * f_1 * g_0 - 4. * f_0 * g_1;
            let v = 2. * j * l + w.pow(2.) - 4. * f_1 * h_0 - 4. * g_1 * g_0 - 4. * h_1 * f_0;
            let u = 2. * w * l - 4. * g_1 * h_0 - 4. * h_1 * g_0;
            let m = l.pow(2.) - 4. * h_1 * h_0;

            let final_val
            = k.pow(4.) * o
            + k.pow(3.) * p
            + k.pow(2.) * v
            + k * u
            + m;

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

#[cfg(test)]
mod tests {
    use nannou::math::num_traits::Pow;

    use crate::utils::deg_to_rad;

    use super::Ellipse;

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

    #[test]
    fn tangent_d_d_0() {
        let k = deg_to_rad(66.).tan();
        let r = E0.tangent_k_alg(&E1, k);
        assert_eq_err!(r.0, 373., 2.);
        assert_eq_err!(r.1, 5., 2.);
    }

    #[test]
    fn tangent_d_d_1() {
        let k = deg_to_rad(50.).tan();
        let r = E0.tangent_k_alg(&E1, k);
        assert_eq_err!(r.0, 9., 2.);
        assert_eq_err!(r.1, 189., 2.);
    }

    #[test]
    fn tangent_d_d_2() {
        let k = deg_to_rad(85.).tan();
        let r = E0.tangent_k_alg(&E1, k);
        assert_eq_err!(r.0, 3300., 2.);
        assert_eq_err!(r.1, 1842., 2.);
    }

    #[test]
    fn tangent_d_d_3() {
        let k = deg_to_rad(25.).tan();
        let r = E0.tangent_k_alg(&E1, k);
        assert_eq_err!(r.0, 263., 2.);
        assert_eq_err!(r.1, 294., 2.);
    }

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
}
