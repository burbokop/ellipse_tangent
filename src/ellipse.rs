use crate::utils::notmalize_array_around_one;
use burbomath::{
    Abs, Cube, Ellipse, FromUSize, IsNeg, IsPositive, Line, Log2, NonNeg, One, Point, Positive,
    SignedSq, SignedSqrt, Sq, Sqrt, TopLimit, Two, UnsignedContstant, Zero, non_neg, uconst,
};
use core::f64;
use num_traits::Pow;
use rustnomial::Polynomial;
use std::{
    iter::Sum,
    ops::{Add, Div, DivAssign, Mul, Neg, Sub},
};

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

pub trait EllipseEx<T> {
    fn common_tangents_intermediate_data(&self, rhs: &Self) -> CommonTangentsIntermediateData<T>
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
        NonNeg<T>: One;

    fn tangent_k_alg(&self, rhs: &Self, k: T) -> (T, T)
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
        NonNeg<T>: Sqrt<Output = NonNeg<T>>;
}

impl<T> EllipseEx<T> for Ellipse<T> {
    fn common_tangents_intermediate_data(&self, rhs: &Self) -> CommonTangentsIntermediateData<T>
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
        let x_0 = self.x().clone();
        let y_0 = self.y().clone();
        let a_0 = self.a().clone();
        let b_0 = self.b().clone();
        let r_0 = self.r().clone();
        let i_0 = self.i().clone();

        let x_1 = rhs.x().clone();
        let y_1 = rhs.y().clone();
        let a_1 = rhs.a().clone();
        let b_1 = rhs.b().clone();
        let r_1 = rhs.r().clone();
        let i_1 = rhs.i().clone();

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

    fn tangent_k_alg(&self, rhs: &Self, k: T) -> (T, T)
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
        let x_0 = self.x().clone();
        let y_0 = self.y().clone();
        let a_0 = self.a().clone();
        let b_0 = self.b().clone();
        let r_0 = self.r().clone();
        let i_0 = self.i().clone();

        let x_1 = rhs.x().clone();
        let y_1 = rhs.y().clone();
        let a_1 = rhs.a().clone();
        let b_1 = rhs.b().clone();
        let r_1 = rhs.r().clone();
        let i_1 = rhs.i().clone();

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

pub trait EllipseExF64 {
    fn common_tangents(&self, rhs: &Self) -> Vec<(Line<f64>, TangentDirection)>;
}

impl EllipseExF64 for Ellipse<f64> {
    fn common_tangents(&self, rhs: &Self) -> Vec<(Line<f64>, TangentDirection)> {
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

/// Into world space
fn into_ws<T>(ellipse: &Ellipse<T>, p: Point<T>) -> Point<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Clone + Zero,
{
    (p.rotated(Point::origin(), ellipse.rotation())).absolute(ellipse.center())
}

/// From world space
fn from_ws<T>(ellipse: &Ellipse<T>, p: Point<T>) -> Point<T>
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
    p.relative(ellipse.center())
        .rotated(Point::origin(), !ellipse.rotation())
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

    static E0: Ellipse<f32> = Ellipse::from_raw(100.0, 100.0, 40.0, 70.0, 0.9659258, 0.25881904);

    static E1: Ellipse<f32> = Ellipse::from_raw(-30.0, -100.0, 20.0, 80.0, 0.50000036, -0.8660252);

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
}
