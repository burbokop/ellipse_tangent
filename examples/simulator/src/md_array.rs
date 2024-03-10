use std::ops::Div;

use nannou::math::num_traits::real::Real;

#[derive(Debug)]
pub struct MdArray<T, const D: usize> {
    data: Vec<T>,
    width: usize,
    height: usize,
}

impl<T, const D: usize> MdArray<T, D> {
    pub fn new(v: T, width: usize, height: usize) -> Self
    where
        T: Clone,
    {
        Self {
            data: vec![v; width * height],
            width,
            height,
        }
    }

    pub fn raw(&self) -> &[T] {
        &self.data
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }

    pub fn at(&self, x: usize, y: usize) -> &T {
        &self.data[y * self.width + x]
    }

    pub fn at_mut(&mut self, x: usize, y: usize) -> &mut T {
        &mut self.data[y * self.width + x]
    }

    pub fn normalized(self) -> Self
    where
        T: Ord,
        T: Div<Output = T>,
        T: Clone,
    {
        let max = self.data.iter().max().unwrap().clone();
        Self {
            data: self.data.into_iter().map(|x| x / max.clone()).collect(),
            width: self.width,
            height: self.height,
        }
    }

    pub fn partially_normalized(self) -> Self
    where
        T: PartialOrd,
        T: Div<Output = T>,
        T: Clone,
        T: Real,
    {
        let max = *self
            .data
            .iter()
            .max_by(|a, b| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        Self {
            data: self.data.into_iter().map(|x| x / max).collect(),
            width: self.width,
            height: self.height,
        }
    }
}
