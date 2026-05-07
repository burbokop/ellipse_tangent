use std::time::Duration;

use burbomath::{Rect, Vector};
use nannou::{
    color::{Rgba8, BLACK, RED, WHITE},
    Draw,
};

use crate::{
    draw::{
        common::draw_vector_with_icon,
        ui::icons::{
            draw_active_maneuver_icon, draw_active_prograde_icon, draw_active_radial_in_icon,
            draw_active_radial_out_icon, draw_active_retrograde_icon, draw_heading_icon,
            draw_maneuver_icon, draw_prograde_icon, draw_radial_in_icon, draw_radial_out_icon,
            draw_retrograde_icon,
        },
    },
    event_handler::AutoRotationTarget,
    palette,
};

pub struct NavCircleData {
    pub prograde: Vector<f32>,
    pub retrograde: Vector<f32>,
    pub radial_in: Vector<f32>,
    pub radial_out: Vector<f32>,
    pub maneuver: Vector<f32>,
    pub auto_rotation_mode: Option<AutoRotationTarget>,
    pub auto_rotation_target: Vector<f32>,
}

pub fn draw_nav_circle(
    draw: &Draw,
    bb: Rect<f32>,
    data: &NavCircleData,
    duration_since_start: Duration,
) {
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

    draw.line()
        .points(
            (*bb.center().x(), *bb.center().y()).into(),
            (
                *bb.center().x() + bb.w() / 8.,
                *bb.center().y() - bb.h() / 8.,
            )
                .into(),
        )
        .color(palette::UI_STROKE_COLOR)
        .weight(1.);

    draw.line()
        .points(
            (*bb.center().x(), *bb.center().y()).into(),
            (
                *bb.center().x() - bb.w() / 8.,
                *bb.center().y() - bb.h() / 8.,
            )
                .into(),
        )
        .color(palette::UI_STROKE_COLOR)
        .weight(1.);

    if data.auto_rotation_mode != None {
        draw_vector_with_icon(
            draw,
            draw_heading_icon,
            center,
            data.auto_rotation_target.norm() * radius,
            palette::UI_STROKE_COLOR,
            1.,
            duration_since_start,
        );
    }

    draw_vector_with_icon(
        draw,
        if data.auto_rotation_mode == Some(AutoRotationTarget::Prograde) {
            draw_active_prograde_icon
        } else {
            draw_prograde_icon
        },
        center,
        data.prograde.norm() * radius,
        palette::PROGRADE_RETROGRADE_COLOR,
        1.,
        duration_since_start,
    );

    draw_vector_with_icon(
        draw,
        if data.auto_rotation_mode == Some(AutoRotationTarget::Retrograde) {
            draw_active_retrograde_icon
        } else {
            draw_retrograde_icon
        },
        center,
        data.retrograde.norm() * radius,
        palette::PROGRADE_RETROGRADE_COLOR,
        1.,
        duration_since_start,
    );

    draw_vector_with_icon(
        draw,
        if data.auto_rotation_mode == Some(AutoRotationTarget::RadialIn) {
            draw_active_radial_in_icon
        } else {
            draw_radial_in_icon
        },
        center,
        data.radial_in.norm() * radius,
        palette::RADIAL_COLOR,
        1.,
        duration_since_start,
    );

    draw_vector_with_icon(
        draw,
        if data.auto_rotation_mode == Some(AutoRotationTarget::RadialOut) {
            draw_active_radial_out_icon
        } else {
            draw_radial_out_icon
        },
        center,
        data.radial_out.norm() * radius,
        palette::RADIAL_COLOR,
        1.,
        duration_since_start,
    );

    draw_vector_with_icon(
        draw,
        if data.auto_rotation_mode == Some(AutoRotationTarget::Maneuver) {
            draw_active_maneuver_icon
        } else {
            draw_maneuver_icon
        },
        center,
        data.maneuver.norm() * radius,
        palette::MANEUVER_COLOR,
        1.,
        duration_since_start,
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

pub struct VesselInfoData {
    pub thrust: f32,
    pub thrust_acceleration: f32,
    pub mass: f32,
    pub todo2: f32,
    pub todo3: f32,
    pub todo4: f32,
}

pub fn draw_vessel_info(draw: &Draw, bb: Rect<f32>, data: &VesselInfoData) {
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
        &format!("Thrust: {:.2} N", data.thrust),
        palette::PROGRADE_RETROGRADE_COLOR,
    );

    draw_text(
        4,
        &format!("Thrust acceleration: {:.2} m/c^2", data.thrust_acceleration),
        palette::PROGRADE_RETROGRADE_COLOR,
    );

    draw_text(
        3,
        &format!("Mass: {:.2} kg", data.mass),
        palette::UI_STROKE_COLOR,
    );

    draw_text(2, &format!("todo2: {:.2}", data.todo2), RED.into());

    draw_text(1, &format!("todo3: {:.2}", data.todo3), RED.into());

    draw_text(0, &format!("todo4: {:.2}", data.todo4), RED.into());
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
            "Turn left/right: A/D",
            "Stop rotation: X",
            "Throttle up/down: W/S",
            "Auto rotate to prograde vel: Alt + Up arrow",
            "Auto rotate to retrograde vel: Alt + Down arrow",
            "Auto rotate to radial out vel: Alt + Left arrow",
            "Auto rotate to radial in vel: Alt + Right arrow",
            "Enter/Exit manuever mode: M",
            "Add manuever prograde vel: Up arrow",
            "Add manuever retrograde vel: Down arrow",
            "Add manuever radial out vel: Left arrow",
            "Add manuever radial in vel: Right arrow",
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

pub struct TimeInfoData {
    pub time_speed: f32,
}

pub fn draw_time_info(draw: &Draw, bb: Rect<f32>, data: &TimeInfoData) {
    draw.rect()
        .x(*bb.center().x())
        .y(*bb.center().y())
        .w(*bb.w())
        .h(*bb.h())
        .color(palette::UI_BACKGROUND_COLOR)
        .stroke_color(palette::UI_STROKE_COLOR)
        .stroke_weight(1.);

    let margin = 8.;

    draw.text(&format!("Time speed: {}X", data.time_speed))
        .x_y(*bb.center().x(), *bb.center().y())
        .w(*bb.w() - 2. * margin)
        .h(*bb.h() - 2. * margin)
        .font_size(14)
        .center_justify()
        .align_text_middle_y()
        .color(palette::MANEUVER_COLOR);
}
