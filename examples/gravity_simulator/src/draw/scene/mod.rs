use crate::{
    draw::scene::{
        celestial_body::draw_celestial_body, orbit::draw_elliptic_orbit, vessel::draw_vessel,
    },
    utils::{color_from_hex, matrix_to_mat4},
    Model, G, PALLETE,
};
use nannou::{color::Rgb, Draw};
use std::time::Duration;

mod celestial_body;
mod orbit;
mod vessel;

pub(crate) fn draw_scene<R: rand::RngCore>(
    draw: &Draw,
    model: &Model<R>,
    duration_since_start: Duration,
) {
    let draw = draw.transform(matrix_to_mat4(model.camera.transformation()));

    draw.background().color(color_from_hex(PALLETE[0]));

    let compensatory_scale = 1. / model.camera.transformation().average_scale();
    let body = model.vessel_orbit.body.upgrade().unwrap();

    draw_celestial_body(&draw, &body, model.vessel_orbit.ellipse.f0());
    draw_elliptic_orbit(
        &draw,
        &model.vessel_orbit,
        Rgb::from_components((0.2, 0.5, 1.)),
        compensatory_scale,
    );
    draw_vessel(
        &draw,
        &model.vessel,
        &model.vessel_orbit,
        G,
        compensatory_scale,
        duration_since_start,
    );

    if model.event_handler_context.manuever_planner_mode() {
        if let Some(manuever) = &model.manuever {
            let manuever_point = model
                .vessel_orbit
                .ellipse
                .point_on_ellipse(manuever.delta_v_anomaly);

            draw.x(*manuever_point.x())
                .y(*manuever_point.y())
                .scale(compensatory_scale)
                .ellipse()
                .radius(4.)
                .color(Rgb::from_components((1., 0.5, 0.3)));

            draw_elliptic_orbit(
                &draw,
                &manuever.orbit,
                Rgb::from_components((1., 0.5, 0.3)),
                compensatory_scale,
            );
        }
    }
}
