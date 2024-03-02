pub fn deg_to_rad(deg: f32) -> f32 {
    deg * std::f32::consts::PI / 180.
}

pub fn deg_to_rot(deg: f32) -> (f32, f32) {
    let rad = deg_to_rad(deg);
    (rad.cos(), rad.sin())
}
