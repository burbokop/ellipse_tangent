use burbomath::Point;
use nannou::Draw;

use crate::orbit::CelestialBody;

pub fn draw_celestial_body(draw: &Draw, body: &CelestialBody, center: Point<f32>) {
    if center.x().is_finite() && center.y().is_finite() {
        assert!(body.atmosphere_radius.0.is_finite());
        assert!(body.solid_radius.0.is_finite());

        draw.ellipse()
            .x(*center.x())
            .y(*center.y())
            .radius(body.atmosphere_radius.0)
            .color(body.atmosphere_color);

        draw.ellipse()
            .x(*center.x())
            .y(*center.y())
            .radius(body.solid_radius.0)
            .color(body.solid_color);
    }
}
