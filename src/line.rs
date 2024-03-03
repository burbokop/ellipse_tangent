use std::f32::EPSILON;

#[derive(Debug, Clone, Copy)]
pub struct Line {
    pub k: f32,
    pub d: f32,
}

impl Line {
    pub fn from_points(x0: f32, y0: f32, x1: f32, y1: f32) -> Option<Self> {
        let dx = x1 - x0;
        if dx.abs() < EPSILON {
            let k = (y1 - y0) / dx;
            let d = y0 - k * x0;
            Some(Self { k, d })
        } else {
            None
        }
    }
}
