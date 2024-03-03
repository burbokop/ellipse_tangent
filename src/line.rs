use std::f32::EPSILON;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Line {
    pub(crate) k: f32,
    pub(crate) d: f32,
}

enum Quadrant {
    LeftTop,
    RightTop,
    RightBottom,
    LeftBottom,
}

impl Line {
    pub(crate) fn from_points(x0: f32, y0: f32, x1: f32, y1: f32) -> Option<Self> {
        // y0 = k * x0 + d;
        // y1 = k * x1 + d;

        // d = y1 - k * x1
        // y0 = k * x0 + y1 - k * x1;

        let dx = x1 - x0;
        if dx.abs() < EPSILON {
            let k = (y1 - y0) / dx;
            let d = y0 - k * x0;
            Some(Self { k, d })
        } else {
            None
        }
    }

    // pub (crate) fn quadrant_of_k(k: f32) -> Quadrant {
    //     if k >= 0 {
    //         if k >= 1 {

    //         } else {

    //         }
    //     } else {
    //         if k < -1 {

    //         } else {

    //         }
    //     }
    // }
}
