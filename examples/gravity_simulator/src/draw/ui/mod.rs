use std::time::Duration;

use burbomath::{Rect, Size};
use nannou::Draw;

use crate::draw::ui::panels::{
    draw_controls_info, draw_flight_info, draw_manuever_info, draw_nav_circle, draw_throttle_bar,
    draw_time_info, draw_vessel_info, ControlsInfoData, FlightInfoData, ManueverInfoData,
    NavCircleData, ThrottleBarData, TimeInfoData, VesselInfoData,
};

pub mod icons;
pub mod panels;

pub struct UIData {
    pub nav_circle: NavCircleData,
    pub flight_info: FlightInfoData,
    pub controls_info: ControlsInfoData,

    pub manuever_info: ManueverInfoData,
    pub throttle_bar: ThrottleBarData,
    pub vessel_info: VesselInfoData,
    pub time_info: TimeInfoData,
}

pub(crate) fn draw_ui(
    draw: &Draw,
    window_rect: Rect<f32>,
    data: &UIData,
    duration_since_start: Duration,
) {
    let throttle_bar_size: Size<_> = (50., 125.).into();
    let throttle_bar_left_margin = 30.;
    let throttle_bar_top_margin = 30.;

    let throttle_bar_bb = (
        window_rect.left() + throttle_bar_left_margin,
        window_rect.top() + throttle_bar_top_margin,
        *throttle_bar_size.w(),
        *throttle_bar_size.h(),
    )
        .into();

    draw_throttle_bar(draw, throttle_bar_bb, &data.throttle_bar);

    let vessel_info_size: Size<_> = (200., 125.).into();
    let vessel_info_left_margin = 30.;
    let vessel_info_top_margin = 30.;

    let vessel_info_bb = (
        window_rect.left()
            + throttle_bar_left_margin
            + throttle_bar_size.w()
            + vessel_info_left_margin,
        window_rect.top() + vessel_info_top_margin,
        *vessel_info_size.w(),
        *vessel_info_size.h(),
    )
        .into();

    draw_vessel_info(draw, vessel_info_bb, &data.vessel_info);

    let nav_circle_size: Size<_> = (125., 125.).into();
    let nav_circle_right_margin = 30.;
    let nav_circle_top_margin = 30.;

    let nav_circle_bb: Rect<f32> = (
        window_rect.right() - nav_circle_size.w() - nav_circle_right_margin,
        window_rect.top() + nav_circle_top_margin,
        *nav_circle_size.w(),
        *nav_circle_size.h(),
    )
        .into();

    draw_nav_circle(
        draw,
        nav_circle_bb.clone(),
        &data.nav_circle,
        duration_since_start,
    );

    let manuever_info_size: Size<_> = (300., 125.).into();
    let manuever_info_right_margin = 30.;
    let manuever_info_top_margin = 30.;

    let manuever_info_bb = (
        window_rect.right()
            - nav_circle_size.w()
            - nav_circle_right_margin
            - manuever_info_size.w()
            - manuever_info_right_margin,
        window_rect.top() + manuever_info_top_margin,
        *manuever_info_size.w(),
        *manuever_info_size.h(),
    )
        .into();

    draw_manuever_info(draw, manuever_info_bb, &data.manuever_info);

    let flight_info_size: Size<_> = (200., 125.).into();
    let flight_info_right_margin = 30.;
    let flight_info_top_margin = 30.;

    let flight_info_bb = (
        window_rect.right()
            - nav_circle_size.w()
            - nav_circle_right_margin
            - manuever_info_size.w()
            - manuever_info_right_margin
            - flight_info_size.w()
            - flight_info_right_margin,
        window_rect.top() + flight_info_top_margin,
        *flight_info_size.w(),
        *flight_info_size.h(),
    )
        .into();

    draw_flight_info(draw, flight_info_bb, &data.flight_info);

    let controls_info_size: Size<_> = (300., 125.).into();
    let controls_info_right_margin = 30.;
    let controls_info_top_margin = 30.;

    let controls_info_bb = (
        window_rect.right()
            - nav_circle_size.w()
            - nav_circle_right_margin
            - manuever_info_size.w()
            - manuever_info_right_margin
            - flight_info_size.w()
            - flight_info_right_margin
            - controls_info_size.w()
            - controls_info_right_margin,
        window_rect.top() + controls_info_top_margin,
        *controls_info_size.w(),
        *controls_info_size.h(),
    )
        .into();

    draw_controls_info(draw, controls_info_bb, &data.controls_info);

    let time_info_size: Size<_> = (125., 40.).into();
    let time_info_right_margin = 30.;
    let time_info_bottom_margin = 30.;

    let time_info_bb = (
        window_rect.right() - time_info_size.w() - time_info_right_margin,
        window_rect.bottom() - time_info_size.h() - time_info_bottom_margin,
        *time_info_size.w(),
        *time_info_size.h(),
    )
        .into();

    draw_time_info(draw, time_info_bb, &data.time_info);
}
