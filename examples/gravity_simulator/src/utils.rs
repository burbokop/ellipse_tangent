use std::{clone, marker::PhantomData, ops::Sub, process::Output};

use burbomath::{camera::Camera, Matrix, Point, Rect};
use nannou::{
    color::Rgba8,
    glam::{Mat3, Mat4},
};

pub(crate) fn matrix_to_mat3(x: Matrix<f32>) -> Mat3 {
    let [a, b, c, d, e, f, g, h, i] = x.into();

    // Mat3::from_cols(
    //     (a,b,c).into(),
    //     (d,e,f).into(),
    //     (g,h,i).into()
    // )

    Mat3::from_cols((a, d, g).into(), (b, e, h).into(), (c, f, i).into())
}

pub(crate) fn matrix_to_mat4(x: Matrix<f32>) -> Mat4 {
    let [a, b, c, d, e, f, g, h, i] = x.into();

    Mat4::from_cols(
        (a, b, 0., c).into(),
        (d, e, 0., f).into(),
        (g, h, i, 0.).into(),
        (0., 0., 0., 1.).into(),
    )
    .transpose()
}

pub(crate) fn nannou_rect_to_rect<T>(rect: nannou::geom::Rect<T>) -> Rect<T>
where
    T: Sub<Output = T> + Clone,
{
    (
        rect.x.start.clone(),
        rect.y.start.clone(),
        rect.x.end - rect.x.start,
        rect.y.end - rect.y.start,
    )
        .into()
}

pub const fn color_from_hex(c: u32) -> Rgba8 {
    let a = (c >> 24) as u8;
    let r = (c >> 16) as u8;
    let g = (c >> 08) as u8;
    let b = (c >> 00) as u8;
    Rgba8 {
        color: nannou::color::rgb::Rgb {
            red: r,
            green: g,
            blue: b,
            standard: PhantomData::default(),
        },
        alpha: a,
    }
}

/// `target_point` - in world coords
/// `view_port_center` - in view port coords
/// Note: wipes rotation. TODO fix it
pub fn center_camera_around_a_point(
    camera: &mut Camera<f32>,
    target_point: Point<f32>,
    view_port_center: Point<f32>,
) {
    let center_tr = Matrix::<f32>::translate(view_port_center - Point::origin());

    let tr = Matrix::<f32>::translate((-target_point.x(), -target_point.y()).into());

    let mut mat = Matrix::identity();

    mat = &mat * &center_tr;
    mat = &mat * camera.scale();
    mat = &mat * camera.rotation();
    mat = &mat * &tr;

    *camera = Camera::default();

    camera.set_translation(Point::origin() + mat.translation());

    camera.set_scale(mat.average_scale());
}
