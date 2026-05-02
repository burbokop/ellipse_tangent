use burbomath::{Rect, Vector};
use nannou::{
    color::{Rgba8, WHITE},
    Draw,
};

use crate::{
    draw::{
        common::draw_vector_with_icon,
        ui::icons::{
            draw_maneuver_icon, draw_prograde_icon, draw_radial_in_icon, draw_radial_out_icon,
            draw_retrograde_icon,
        },
    },
    palette,
};

pub struct NavCircleData {
    pub prograde: Vector<f32>,
    pub retrograde: Vector<f32>,
    pub radial_in: Vector<f32>,
    pub radial_out: Vector<f32>,
    pub maneuver: Vector<f32>,
}

pub fn draw_nav_circle(draw: &Draw, bb: Rect<f32>, data: &NavCircleData) {
    let center = bb.center();

    draw.ellipse()
        .x(*center.x())
        .y(*center.y())
        .w(*bb.w())
        .h(*bb.h())
        .color(palette::UI_BACKGROUND_COLOR)
        .stroke_color(palette::UI_STROKE_COLOR)
        .stroke_weight(1.);

    let radius = f32::min(*bb.w(), *bb.h()) / 2.;

    draw.line()
        .points(
            (bb.left(), *bb.center().y()).into(),
            (bb.right(), *bb.center().y()).into(),
        )
        .color(palette::UI_STROKE_COLOR)
        .weight(1.);

    draw.line()
        .points(
            (*bb.center().x(), bb.top()).into(),
            (*bb.center().x(), bb.bottom()).into(),
        )
        .color(palette::UI_STROKE_COLOR)
        .weight(1.);

    draw_vector_with_icon(
        draw,
        draw_prograde_icon,
        center,
        data.prograde.norm() * radius,
        palette::PROGRADE_RETROGRADE_COLOR,
        1.,
    );

    draw_vector_with_icon(
        draw,
        draw_retrograde_icon,
        center,
        data.retrograde.norm() * radius,
        palette::PROGRADE_RETROGRADE_COLOR,
        1.,
    );

    draw_vector_with_icon(
        draw,
        draw_radial_in_icon,
        center,
        data.radial_in.norm() * radius,
        palette::RADIAL_COLOR,
        1.,
    );

    draw_vector_with_icon(
        draw,
        draw_radial_out_icon,
        center,
        data.radial_out.norm() * radius,
        palette::RADIAL_COLOR,
        1.,
    );

    draw_vector_with_icon(
        draw,
        draw_maneuver_icon,
        center,
        data.maneuver.norm() * radius,
        palette::MANEUVER_COLOR,
        1.,
    );
}

pub struct FlightInfoData {
    pub velocity: f32,
    pub apoapsis: f32,
    pub periapsis: f32,
    pub delta_v_capacity: f32,
    pub delta_v_needed_for_manuever: f32,
    pub time_to_next_transition_point: f32,
}

pub fn draw_flight_info(draw: &Draw, bb: Rect<f32>, data: &FlightInfoData) {
    draw.rect()
        .x(*bb.center().x())
        .y(*bb.center().y())
        .w(*bb.w())
        .h(*bb.h())
        .color(palette::UI_BACKGROUND_COLOR)
        .stroke_color(palette::UI_STROKE_COLOR)
        .stroke_weight(1.);

    let left_margin = 8.;
    let row_count = 6;

    let row_height = bb.h() / row_count as f32;
    let draw_text = |index: usize, text: &str, color: Rgba8| {
        draw.text(text)
            .x_y(
                left_margin + *bb.center().x(),
                bb.y() + row_height / 2. + row_height * index as f32,
            )
            .w(*bb.w())
            .h(row_height)
            .font_size(12)
            .left_justify()
            .align_text_middle_y()
            .color(color);
    };

    draw_text(
        5,
        &format!("Velocity: {:.2} m/s", data.velocity),
        palette::PROGRADE_RETROGRADE_COLOR,
    );

    draw_text(
        4,
        &format!("Apoapsis: {:.2} m", data.apoapsis),
        palette::PROGRADE_RETROGRADE_COLOR,
    );

    draw_text(
        3,
        &format!("Periapsis: {:.2} m", data.periapsis),
        palette::PROGRADE_RETROGRADE_COLOR,
    );

    draw_text(
        2,
        &format!("Δv capacity: {:.2} m/s", data.delta_v_capacity),
        palette::PROGRADE_RETROGRADE_COLOR,
    );

    draw_text(
        1,
        &format!("Manuever Δv: {:.2} m/s", data.delta_v_needed_for_manuever),
        palette::MANEUVER_COLOR,
    );

    draw_text(
        0,
        &format!("Time to trans: {:.2} s", data.time_to_next_transition_point),
        palette::MANEUVER_COLOR,
    );
}

pub fn draw_controls_info(draw: &Draw, bb: Rect<f32>) {
    draw.rect()
        .x(*bb.center().x())
        .y(*bb.center().y())
        .w(*bb.w())
        .h(*bb.h())
        .color(palette::UI_BACKGROUND_COLOR)
        .stroke_color(palette::UI_STROKE_COLOR)
        .stroke_weight(1.);

    let margin = 8.;

    draw.text(
        &[
            "Move camera vertically: Wheel",
            "Move camera horisontally: Shift + Wheel",
            "Zoom in/out: Ctrl + Wheel",
            "Center on vessel: C",
            "Turn left: A",
            "Turn right: D",
            "Throttle up: W",
            "Throttle down: S",
            "Enter/Exit manuever mode: M",
            "Add manuever prograde vel: Up arrow",
            "Add manuever retrograde vel: Down arrow",
            "Add manuever radial in vel: Left arrow",
            "Add manuever radial out vel: Right arrow",
        ]
        .join("\n"),
    )
    .x_y(*bb.center().x(), *bb.center().y())
    .w(*bb.w() - 2. * margin)
    .h(*bb.h() - 2. * margin)
    .font_size(10)
    .left_justify()
    .align_text_middle_y()
    .color(WHITE);
}

pub struct ManueverInfoData {
    pub manuever_mode: bool,
}

pub fn draw_manuever_info(draw: &Draw, bb: Rect<f32>, data: &ManueverInfoData) {
    draw.rect()
        .x(*bb.center().x())
        .y(*bb.center().y())
        .w(*bb.w())
        .h(*bb.h())
        .color(palette::UI_BACKGROUND_COLOR)
        .stroke_color(palette::UI_STROKE_COLOR)
        .stroke_weight(1.);

    let margin = 8.;

    draw.text(if data.manuever_mode {
        "Manuever mode ON"
    } else {
        "Manuever mode OFF"
    })
    .x_y(*bb.center().x(), *bb.center().y())
    .w(*bb.w() - 2. * margin)
    .h(*bb.h() - 2. * margin)
    .font_size(20)
    .center_justify()
    .align_text_middle_y()
    .color(palette::MANEUVER_COLOR);
}

pub struct ThrottleBarData {
    pub throttle: f32,
}

pub fn draw_throttle_bar(draw: &Draw, bb: Rect<f32>, data: &ThrottleBarData) {
    draw.rect()
        .x(*bb.center().x())
        .y(*bb.center().y())
        .w(*bb.w())
        .h(*bb.h())
        .color(palette::UI_BACKGROUND_COLOR)
        .stroke_color(palette::UI_STROKE_COLOR)
        .stroke_weight(1.);

    let margin = 8.;

    draw.rect()
        .x(*bb.center().x())
        .y(*bb.center().y())
        .w(*bb.w() - 2. * margin)
        .h(*bb.h() - 2. * margin)
        .no_fill()
        .stroke_color(palette::UI_STROKE_COLOR)
        .stroke_weight(1.);

    let fill_bb: Rect<_> = (
        *bb.x() + margin + 1.,
        *bb.y() + margin + 1.,
        *bb.w() - 2. * margin - 2.,
        (*bb.h() - 2. * margin - 2.) * data.throttle,
    )
        .into();

    draw.rect()
        .x(*fill_bb.center().x())
        .y(*fill_bb.center().y())
        .w(*fill_bb.w())
        .h(*fill_bb.h())
        .color(palette::PROGRADE_RETROGRADE_COLOR);
}
