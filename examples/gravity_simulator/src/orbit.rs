use std::{rc::Weak, time::Duration};

use burbomath::{physics::Kg, Angle, Vector};
use ellipse_tangent::ellipse::Ellipse;

use crate::G;

pub struct CelestialBody {
    pub mass: Kg<f32>,
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
