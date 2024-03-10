use std::f32::EPSILON;

#[derive(Debug, Clone, Copy)]
pub struct SimpleLine {
    pub k: f32,
    pub d: f32,
}

impl SimpleLine {
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

// Line equesion in complex numbers
// ax + by + c = 0;
// by = -ax -c
// y = (-ax - c) / b
// y = -ax / b - c / b
// y = x (-a / b) - c / b
//
// k = -a / b
// d = -c / b
//
// tan = -a / b
// sin = -a
// cos = b
//
// a = -sin
// b = cos
//
// d = -c / cos
//
// c = -d cos
//
//trigonometric result: -sinθ x + cosθ y - d cosθ = 0;
//
//complex result: -ix + ry - dr = 0;
pub struct Line {
    pub r: f32,
    pub i: f32,
    pub d: f32
}

impl Line {
    pub fn from_angle(theta: f32, d: f32) -> Self {
        Line { r: theta.cos(), i: theta.sin(), d }
    }

    pub fn from_points(x0: f32, y0: f32, x1: f32, y1: f32) -> Option<Self> {
        let dx = x1 - x0;
        if dx.abs() < EPSILON {
            let k = (y1 - y0) / dx;
            let d = y0 - k * x0;

            // TODO solve equesion system
            //-ix_0 + ry_0 - dr = 0
            //-ix_1 + ry_1 - dr = 0
            // i^2 + r^2 = 1
            todo!()
        } else {
            None
        }
    }
}
