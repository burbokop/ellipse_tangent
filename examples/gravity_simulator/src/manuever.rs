use std::time::Duration;

use crate::orbit::EllipticOrbit;
use burbomath::{Angle, DeltaAngle, Vector};

const ANOMALY_CHANGE_SPEED_FACTOR: f32 = 1.;
const VELOCITY_CHANGE_SPEED_FACTOR: f32 = 0.1;

#[derive(Debug)]
pub struct Manuever {
    pub orbit: EllipticOrbit,
    pub relative_delta_v: Vector<f32>,
    pub delta_v_anomaly: Angle<f32>,
}

impl Manuever {
    pub fn delta_v(
        &self,
        initial_orbit: &EllipticOrbit,
        gravitational_constant: f32,
    ) -> Vector<f32> {
        let body = initial_orbit.body.upgrade().unwrap();
        let tangential_velocity = initial_orbit
            .ellipse
            .tangential_velocity(self.delta_v_anomaly, body.mass, gravitational_constant)
            .norm();

        self.relative_delta_v * tangential_velocity.rotor()
    }

    pub fn move_start_anomaly_forward(
        &mut self,
        initial_orbit: &EllipticOrbit,
        dt: Duration,
        gravitational_constant: f32,
    ) {
        self.delta_v_anomaly +=
            DeltaAngle::from_radians(ANOMALY_CHANGE_SPEED_FACTOR) * dt.as_secs_f32();

        self.orbit = initial_orbit.accelerated_at_anomaly(
            self.delta_v(initial_orbit, gravitational_constant),
            self.delta_v_anomaly,
            gravitational_constant,
        );
    }

    pub fn move_start_anomaly_backward(
        &mut self,
        initial_orbit: &EllipticOrbit,
        dt: Duration,
        gravitational_constant: f32,
    ) {
        self.delta_v_anomaly -=
            DeltaAngle::from_radians(ANOMALY_CHANGE_SPEED_FACTOR) * dt.as_secs_f32();

        self.orbit = initial_orbit.accelerated_at_anomaly(
            self.delta_v(initial_orbit, gravitational_constant),
            self.delta_v_anomaly,
            gravitational_constant,
        );
    }

    pub fn accelerate_towards_prograde(
        &mut self,
        initial_orbit: &EllipticOrbit,
        dt: Duration,
        gravitational_constant: f32,
    ) {
        let body = initial_orbit.body.upgrade().unwrap();
        let tangential_velocity = initial_orbit.ellipse.tangential_velocity(
            self.delta_v_anomaly,
            body.mass,
            gravitational_constant,
        );

        self.relative_delta_v += Vector::from((1., 0.))
            * tangential_velocity.len()
            * VELOCITY_CHANGE_SPEED_FACTOR
            * dt.as_secs_f32();

        self.orbit = initial_orbit.accelerated_at_anomaly(
            self.delta_v(initial_orbit, gravitational_constant),
            self.delta_v_anomaly,
            gravitational_constant,
        );
    }

    pub fn accelerate_towards_retrograde(
        &mut self,
        initial_orbit: &EllipticOrbit,
        dt: Duration,
        gravitational_constant: f32,
    ) {
        let body = initial_orbit.body.upgrade().unwrap();
        let tangential_velocity = initial_orbit.ellipse.tangential_velocity(
            self.delta_v_anomaly,
            body.mass,
            gravitational_constant,
        );

        self.relative_delta_v += Vector::from((-1., 0.))
            * tangential_velocity.len()
            * VELOCITY_CHANGE_SPEED_FACTOR
            * dt.as_secs_f32();

        self.orbit = initial_orbit.accelerated_at_anomaly(
            self.delta_v(initial_orbit, gravitational_constant),
            self.delta_v_anomaly,
            gravitational_constant,
        );
    }

    pub fn accelerate_towards_radial_in(
        &mut self,
        initial_orbit: &EllipticOrbit,
        dt: Duration,
        gravitational_constant: f32,
    ) {
        let body = initial_orbit.body.upgrade().unwrap();
        let tangential_velocity = initial_orbit.ellipse.tangential_velocity(
            self.delta_v_anomaly,
            body.mass,
            gravitational_constant,
        );

        self.relative_delta_v += Vector::from((0., -1.))
            * tangential_velocity.len()
            * VELOCITY_CHANGE_SPEED_FACTOR
            * dt.as_secs_f32();

        self.orbit = initial_orbit.accelerated_at_anomaly(
            self.delta_v(initial_orbit, gravitational_constant),
            self.delta_v_anomaly,
            gravitational_constant,
        );
    }

    pub fn accelerate_towards_radial_out(
        &mut self,
        initial_orbit: &EllipticOrbit,
        dt: Duration,
        gravitational_constant: f32,
    ) {
        let body = initial_orbit.body.upgrade().unwrap();
        let tangential_velocity = initial_orbit.ellipse.tangential_velocity(
            self.delta_v_anomaly,
            body.mass,
            gravitational_constant,
        );

        self.relative_delta_v += Vector::from((0., 1.))
            * tangential_velocity.len()
            * VELOCITY_CHANGE_SPEED_FACTOR
            * dt.as_secs_f32();

        self.orbit = initial_orbit.accelerated_at_anomaly(
            self.delta_v(initial_orbit, gravitational_constant),
            self.delta_v_anomaly,
            gravitational_constant,
        );
    }
}
