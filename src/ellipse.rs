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

        let discriminant = 4. * (k * x_0 - y_0).pow(2.)
            - 4. * (k.pow(2.) * x_0.pow(2.) - 2. * k * x_0 * y_0
                + y_0.pow(2.)
                + (a.pow(2.) * (-k.pow(2.) * r.pow(2.) - 2. * i * k * r - i.pow(2.))
                    + b.pow(2.) * (-i.pow(2.) * k.pow(2.) + 2. * i * k * r - r.pow(2.)))
                    / (r.pow(4.) + 2. * i.pow(2.) * r.pow(2.) + i.pow(4.)));

        let d_0 = (-2. * (k * x_0 - y_0) + discriminant.sqrt()) / 2.;
        let d_1 = (-2. * (k * x_0 - y_0) - discriminant.sqrt()) / 2.;

        (d_0, d_1)
    }
}
