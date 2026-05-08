use core::f32;
use std::{rc::Weak, time::Duration};

use burbomath::{
    physics::{Kg, KgPerM3, M, M3},
    Angle, Vector,
};
use ellipse_tangent::ellipse::Ellipse;
use nannou::color::Rgba8;

use crate::G;

#[derive(Debug)]
pub struct CelestialBody {
    pub mass: Kg<f32>,
    pub solid_radius: M<f32>,
    pub atmosphere_radius: M<f32>,
    pub solid_color: Rgba8,
    pub atmosphere_color: Rgba8,
}

impl CelestialBody {
    pub fn from_density(
        density: KgPerM3<f32>,
        solid_radius: M<f32>,
        atmosphere_radius: M<f32>,
        solid_color: Rgba8,
        atmosphere_color: Rgba8,
    ) -> Self {
        let volume = solid_radius.cube() * (4. / 3. * f32::consts::PI);
        let mass = volume * density;
        Self {
            mass,
            solid_radius,
            atmosphere_radius,
            solid_color,
            atmosphere_color,
        }
    }
}

pub struct EllipticOrbit {
    pub body: Weak<CelestialBody>,
    pub ellipse: Ellipse,
    pub anomaly: Angle<f32>,
}

impl EllipticOrbit {
    pub fn accelerate(&mut self, acceleration: Vector<f32>, dt: Duration) {
        let body = self.body.upgrade().unwrap();
        self.ellipse =
            self.ellipse
                .accelerated(self.anomaly, body.mass.clone(), G, dt, acceleration);
    }

    pub fn proceed(&mut self, dt: Duration) {
        let body = self.body.upgrade().unwrap();

        let angular_velocity = self
            .ellipse
            .angular_velocity(self.anomaly, body.mass.clone(), G);

        self.anomaly += angular_velocity * dt.as_secs_f32();
    }
}
