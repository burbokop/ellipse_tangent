use core::f32;
use std::{rc::Weak, time::Duration};

use burbomath::{
    physics::{Kg, KgPerM3, M},
    time::RelativeDuration,
    Angle, Ellipse, Vector,
};
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

// /// The angle between x axis and vector from
// struct WorldAnomaly<T>(Angle<T>);
// struct TrueAnomaly<T>(Angle<T>);

#[derive(Debug, Clone)]
pub struct EllipticOrbit {
    pub body: Weak<CelestialBody>,
    pub ellipse: Ellipse<f32>,
    pub anomaly: Angle<f32>,
}

impl EllipticOrbit {
    pub fn time_to(&self, anomaly: Angle<f32>) -> RelativeDuration {
        let body = self.body.upgrade().unwrap();
        self.ellipse
            .relative_time_between_anomalies(self.anomaly, anomaly, body.mass, G)
    }

    /// Changes orbit and returns delta velocity
    pub fn accelerate(&mut self, acceleration: Vector<f32>, dt: Duration) -> Vector<f32> {
        let body = self.body.upgrade().unwrap();

        let position = self.ellipse.point_on_ellipse(self.anomaly);

        self.ellipse =
            self.ellipse
                .accelerated(self.anomaly, body.mass.clone(), G, dt, acceleration);

        self.anomaly = self.ellipse.anomaly(position);

        acceleration * dt.as_secs_f32()
    }

    pub fn proceed(&mut self, dt: Duration) {
        let body = self.body.upgrade().unwrap();

        let angular_velocity = self
            .ellipse
            .angular_velocity(self.anomaly, body.mass.clone(), G)
            .unwrap();

        self.anomaly += angular_velocity * dt.as_secs_f32();
    }

    pub fn accelerated_at_anomaly(
        &self,
        delta_v: Vector<f32>,
        delta_v_anomaly: Angle<f32>,
        gravitational_constant: f32,
    ) -> EllipticOrbit {
        let body = self.body.upgrade().unwrap();

        let f0 = self.ellipse.f0();
        let p = self.ellipse.point_on_ellipse(delta_v_anomaly);

        let vel = self
            .ellipse
            .tangential_velocity(delta_v_anomaly, body.mass.clone(), gravitational_constant)
            .unwrap();

        let (_excentricity, new_f1) = self.ellipse.f1_from_tangential_velocity(
            delta_v_anomaly,
            body.mass.clone(),
            gravitational_constant,
            vel + delta_v,
        );

        let new_ellipse = Ellipse::from_foci(f0, new_f1, p);
        let new_anomaly = Angle::from_radians(
            delta_v_anomaly.radians().into_inner() / self.ellipse.perimeter()
                * new_ellipse.perimeter(),
        );

        EllipticOrbit {
            body: self.body.clone(),
            ellipse: new_ellipse,
            anomaly: new_anomaly,
        }
    }
}
