use std::time::Duration;

use burbomath::{Angle, Complex, DeltaAngle, NonNeg, Vector, Zero};
use nannou::math::{partial_max, partial_min};

pub struct ThrustControl {
    thrust: NonNeg<f32>,
    thrust_change_velocity: f32,
    max_thrust: NonNeg<f32>,
    max_thrust_change_velocity: NonNeg<f32>,
    thrust_change_acceleration: NonNeg<f32>,
}

impl ThrustControl {
    fn new(
        max_thrust: NonNeg<f32>,
        thrust_change_acceleration: NonNeg<f32>,
        max_thrust_change_velocity: NonNeg<f32>,
    ) -> Self {
        Self {
            thrust: NonNeg::zero(),
            thrust_change_velocity: 0.,
            max_thrust,
            max_thrust_change_velocity,
            thrust_change_acceleration,
        }
    }

    pub(crate) fn thrust_up(&mut self, dt: Duration) {
        self.thrust_change_velocity = (self.thrust_change_velocity
            + self.thrust_change_acceleration.into_inner() * dt.as_secs_f32())
        .min(self.max_thrust_change_velocity.into_inner())
        .max(-self.max_thrust_change_velocity.into_inner());
    }

    pub(crate) fn thrust_down(&mut self, dt: Duration) {
        self.thrust_change_velocity = (self.thrust_change_velocity
            - self.thrust_change_acceleration.into_inner() * dt.as_secs_f32())
        .min(self.max_thrust_change_velocity.into_inner())
        .max(-self.max_thrust_change_velocity.into_inner());
    }

    pub(crate) fn brake_thrust_change(&mut self, dt: Duration) {
        if self.thrust_change_velocity >= 0. {
            self.thrust_change_velocity = (self.thrust_change_velocity
                - self.thrust_change_acceleration.into_inner() * dt.as_secs_f32())
            .max(0.);
        } else {
            self.thrust_change_velocity = (self.thrust_change_velocity
                + self.thrust_change_acceleration.into_inner() * dt.as_secs_f32())
            .min(0.);
        }
    }

    pub(crate) fn proceed(&mut self, dt: Duration) {
        self.thrust = NonNeg::new(
            (self.thrust.into_inner() + self.thrust_change_velocity * dt.as_secs_f32())
                .min(self.max_thrust.into_inner())
                .max(0.),
        )
        .unwrap();
    }

    pub(crate) fn thrust(&self) -> NonNeg<f32> {
        self.thrust
    }

    pub(crate) fn max_thrust(&self) -> NonNeg<f32> {
        self.max_thrust
    }
}

pub struct KinematicBody {
    rotation: Angle<f32>,
    rotation_velocity: DeltaAngle<f32>,
    rotation_acceleration: DeltaAngle<f32>,
    thrust_control: ThrustControl,
    mass: NonNeg<f32>,
}

pub struct Vessel {
    pub kinematic_body: KinematicBody,
}

impl KinematicBody {
    pub(crate) fn new(
        rotation_acceleration: DeltaAngle<f32>,
        max_thrust: NonNeg<f32>,
        thrust_change_acceleration: NonNeg<f32>,
        max_thrust_change_velocity: NonNeg<f32>,
        mass: NonNeg<f32>,
    ) -> Self {
        Self {
            rotation: Angle::from_radians(0.),
            rotation_velocity: DeltaAngle::from_radians(0.),
            rotation_acceleration,
            thrust_control: ThrustControl::new(
                max_thrust,
                thrust_change_acceleration,
                max_thrust_change_velocity,
            ),
            mass,
        }
    }

    pub(crate) fn rotation(&self) -> Angle<f32> {
        self.rotation
    }

    pub(crate) fn rotation_velocity(&self) -> DeltaAngle<f32> {
        self.rotation_velocity
    }

    pub(crate) fn rotation_acceleration(&self) -> DeltaAngle<f32> {
        self.rotation_acceleration
    }

    pub(crate) fn thrust(&self) -> NonNeg<f32> {
        self.thrust_control.thrust()
    }

    pub(crate) fn max_thrust(&self) -> NonNeg<f32> {
        self.thrust_control.max_thrust()
    }

    pub(crate) fn mass(&self) -> NonNeg<f32> {
        self.mass
    }

    pub(crate) fn acceleration(&self) -> NonNeg<f32> {
        self.thrust() / self.mass
    }

    pub(crate) fn heading(&self) -> Vector<f32> {
        Vector::from_polar(1., self.rotation)
    }

    pub(crate) fn complex_heading(&self) -> Complex<f32> {
        Complex::from_polar(1., self.rotation)
    }

    pub(crate) fn rotate_left(&mut self, dt: Duration) {
        self.rotation_velocity += self.rotation_acceleration * dt.as_secs_f32();
    }

    pub(crate) fn rotate_right(&mut self, dt: Duration) {
        self.rotation_velocity -= self.rotation_acceleration * dt.as_secs_f32();
    }

    pub(crate) fn brake_rotation(&mut self, dt: Duration) {
        if self.rotation_velocity.radians() >= 0. {
            self.rotation_velocity = partial_max(
                self.rotation_velocity - self.rotation_acceleration * dt.as_secs_f32(),
                DeltaAngle::from_radians(0.),
            );
        } else {
            self.rotation_velocity = partial_min(
                self.rotation_velocity + self.rotation_acceleration * dt.as_secs_f32(),
                DeltaAngle::from_radians(0.),
            );
        }
    }

    pub(crate) fn thrust_up(&mut self, dt: Duration) {
        self.thrust_control.thrust_up(dt);
    }

    pub(crate) fn thrust_down(&mut self, dt: Duration) {
        self.thrust_control.thrust_down(dt);
    }

    pub(crate) fn proceed(&mut self, dt: Duration) {
        self.rotation += self.rotation_velocity * dt.as_secs_f32();
        self.thrust_control.proceed(dt);
    }

    pub(crate) fn brake_thrust_change(&mut self, dt: Duration) {
        self.thrust_control.brake_thrust_change(dt);
    }
}
