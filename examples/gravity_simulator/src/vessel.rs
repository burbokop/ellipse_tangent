use burbomath::{Angle, Complex, DeltaAngle, Vector};

pub struct KinematicBody {
    rotation: Angle<f32>,
    rotation_velocity: DeltaAngle<f32>,
    rotation_acceleration: DeltaAngle<f32>,
    thrust: f32,
    max_thrust: f32,
    mass: f32,
}

pub struct Vessel {
    pub kinematic_body: KinematicBody,
}

impl KinematicBody {
    pub(crate) fn new(rotation_acceleration: DeltaAngle<f32>, max_thrust: f32, mass: f32) -> Self {
        Self {
            rotation: Angle::from_radians(0.),
            rotation_velocity: DeltaAngle::from_radians(0.),
            rotation_acceleration,
            thrust: 0.,
            max_thrust,
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

    pub(crate) fn thrust(&self) -> f32 {
        self.thrust
    }

    pub(crate) fn max_thrust(&self) -> f32 {
        self.max_thrust
    }

    pub(crate) fn mass(&self) -> f32 {
        self.mass
    }

    pub(crate) fn acceleration(&self) -> f32 {
        self.thrust / self.mass
    }

    pub(crate) fn heading(&self) -> Vector<f32> {
        Vector::from_polar(1., self.rotation)
    }

    pub(crate) fn complex_heading(&self) -> Complex<f32> {
        Complex::from_polar(1., self.rotation)
    }

    pub(crate) fn rotate_left(&mut self) {
        todo!()
    }

    pub(crate) fn rotate_right(&mut self) {
        todo!()
    }

    pub(crate) fn brake_rotation(&mut self) {
        todo!()
    }

    pub(crate) fn thrust_up(&mut self) {
        todo!()
    }

    pub(crate) fn thrust_down(&mut self) {
        todo!()
    }
}
