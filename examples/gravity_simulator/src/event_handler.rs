use crate::manuever::Manuever;
use crate::Model;
use burbomath::Point;
use burbomath::Vector;
use nannou::event;
use nannou::App;
use nannou::{
    image::{DynamicImage, RgbaImage},
    prelude::*,
    text::Font,
};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum AutoRotationTarget {
    Prograde,
    Retrograde,
    RadialIn,
    RadialOut,
    Maneuver,
}

pub struct EventHandlerContext {
    control: bool,
    shift: bool,
    alt: bool,
    mouse_position: Point<i32>,
    mouse_position_in_world_space: Point<f32>,
    center_on_vessel_mode: bool,
    auto_rotation_mode: Option<AutoRotationTarget>,
    manuever_planner_mode: bool,
    w_pressed: bool,
    a_pressed: bool,
    s_pressed: bool,
    d_pressed: bool,
    x_pressed: bool,
    less_pressed: bool,
    greater_pressed: bool,
    left_arrow_pressed: bool,
    right_arrow_pressed: bool,
    up_arrow_pressed: bool,
    down_arrow_pressed: bool,
}

impl Default for EventHandlerContext {
    fn default() -> Self {
        Self {
            control: false,
            shift: false,
            alt: false,
            mouse_position: (0, 0).into(),
            mouse_position_in_world_space: (0., 0.).into(),
            center_on_vessel_mode: true,
            auto_rotation_mode: None,
            manuever_planner_mode: false,
            w_pressed: false,
            a_pressed: false,
            s_pressed: false,
            d_pressed: false,
            x_pressed: false,
            less_pressed: false,
            greater_pressed: false,
            left_arrow_pressed: false,
            right_arrow_pressed: false,
            up_arrow_pressed: false,
            down_arrow_pressed: false,
        }
    }
}

impl EventHandlerContext {
    pub fn center_on_vessel_mode(&self) -> bool {
        self.center_on_vessel_mode
    }

    pub fn auto_rotation_mode(&self) -> Option<AutoRotationTarget> {
        self.auto_rotation_mode
    }

    pub fn manuever_planner_mode(&self) -> bool {
        self.manuever_planner_mode
    }

    pub fn w_pressed(&self) -> bool {
        self.w_pressed
    }

    pub fn a_pressed(&self) -> bool {
        self.a_pressed
    }

    pub fn s_pressed(&self) -> bool {
        self.s_pressed
    }

    pub fn d_pressed(&self) -> bool {
        self.d_pressed
    }

    pub fn x_pressed(&self) -> bool {
        self.x_pressed
    }

    pub fn less_pressed(&self) -> bool {
        self.less_pressed
    }

    pub fn greater_pressed(&self) -> bool {
        self.greater_pressed
    }

    pub fn left_arrow_pressed(&self) -> bool {
        self.left_arrow_pressed
    }

    pub fn right_arrow_pressed(&self) -> bool {
        self.right_arrow_pressed
    }

    pub fn up_arrow_pressed(&self) -> bool {
        self.up_arrow_pressed
    }

    pub fn down_arrow_pressed(&self) -> bool {
        self.down_arrow_pressed
    }
}

pub fn event<R: rand::RngCore>(app: &App, model: &mut Model<R>, event: Event) {
    match event {
        Event::WindowEvent { id, simple } => {
            let _window = app.window(id).unwrap();
            let ctx = &mut model.event_handler_context;

            match simple {
                Some(e) => match e {
                    Moved(_vec2) => todo!(),
                    KeyPressed(Key::LShift | Key::RShift) => ctx.shift = true,
                    KeyReleased(Key::LShift | Key::RShift) => ctx.shift = false,
                    KeyPressed(Key::LControl | Key::RControl) => ctx.control = true,
                    KeyReleased(Key::LControl | Key::RControl) => ctx.control = false,
                    KeyPressed(Key::LAlt | Key::RAlt) => ctx.alt = true,
                    KeyReleased(Key::LAlt | Key::RAlt) => ctx.alt = false,
                    KeyReleased(Key::C) => ctx.center_on_vessel_mode = true,
                    KeyPressed(Key::Up) if ctx.alt => {
                        ctx.auto_rotation_mode = Some(AutoRotationTarget::Prograde)
                    }
                    KeyPressed(Key::Up) => ctx.up_arrow_pressed = true,
                    KeyReleased(Key::Up) => ctx.up_arrow_pressed = false,
                    KeyPressed(Key::Down) if ctx.alt => {
                        ctx.auto_rotation_mode = Some(AutoRotationTarget::Retrograde)
                    }
                    KeyPressed(Key::Down) => ctx.down_arrow_pressed = true,
                    KeyReleased(Key::Down) => ctx.down_arrow_pressed = false,
                    KeyPressed(Key::Left) if ctx.alt => {
                        ctx.auto_rotation_mode = Some(AutoRotationTarget::RadialOut)
                    }
                    KeyPressed(Key::Left) => ctx.left_arrow_pressed = true,
                    KeyReleased(Key::Left) => ctx.left_arrow_pressed = false,
                    KeyPressed(Key::Right) if ctx.alt => {
                        ctx.auto_rotation_mode = Some(AutoRotationTarget::RadialIn)
                    }
                    KeyPressed(Key::Right) => ctx.right_arrow_pressed = true,
                    KeyReleased(Key::Right) => ctx.right_arrow_pressed = false,
                    KeyPressed(Key::M) if ctx.alt && ctx.manuever_planner_mode => {
                        ctx.auto_rotation_mode = Some(AutoRotationTarget::Maneuver)
                    }
                    KeyReleased(Key::M) if !ctx.alt => {
                        ctx.manuever_planner_mode = !ctx.manuever_planner_mode;
                        if ctx.manuever_planner_mode && model.manuever.is_none() {
                            model.manuever = Some(Manuever {
                                orbit: model.vessel_orbit.clone(),
                                delta_v: (0., 0.).into(),
                                delta_v_anomaly: model.vessel_orbit.anomaly,
                            })
                        }
                    }
                    KeyPressed(Key::W) => ctx.w_pressed = true,
                    KeyReleased(Key::W) => ctx.w_pressed = false,
                    KeyPressed(Key::A) => {
                        ctx.a_pressed = true;
                        ctx.auto_rotation_mode = None
                    }
                    KeyReleased(Key::A) => ctx.a_pressed = false,
                    KeyPressed(Key::S) => ctx.s_pressed = true,
                    KeyReleased(Key::S) => ctx.s_pressed = false,
                    KeyPressed(Key::D) => {
                        ctx.d_pressed = true;
                        ctx.auto_rotation_mode = None
                    }
                    KeyReleased(Key::D) => ctx.d_pressed = false,
                    KeyPressed(Key::X) => {
                        ctx.x_pressed = true;
                        ctx.auto_rotation_mode = None
                    }
                    KeyReleased(Key::X) => ctx.x_pressed = false,
                    KeyPressed(Key::Comma) => ctx.less_pressed = true,
                    KeyReleased(Key::Comma) => ctx.less_pressed = false,
                    KeyPressed(Key::Period) => ctx.greater_pressed = true,
                    KeyReleased(Key::Period) => ctx.greater_pressed = false,
                    KeyReleased(Key::Key1) => model.time_speed = 1.,
                    KeyReleased(Key::Key2) => model.time_speed = 2.,
                    KeyReleased(Key::Key3) => model.time_speed = 4.,
                    KeyReleased(Key::Key4) => model.time_speed = 8.,
                    KeyReleased(Key::Key5) => model.time_speed = 16.,
                    KeyReleased(Key::Key6) => model.time_speed = 32.,
                    KeyReleased(Key::Key7) => model.time_speed = 64.,
                    KeyReleased(Key::Key8) => model.time_speed = 128.,
                    KeyReleased(Key::Key9) => model.time_speed = 256.,
                    KeyPressed(..) => {}
                    KeyReleased(k) => {
                        println!("Key released: {:?}", k)
                    }
                    ReceivedCharacter(_) => {}
                    MouseMoved(vec2) => {
                        ctx.mouse_position = (vec2.x as i32, vec2.y as i32).into();
                        ctx.mouse_position_in_world_space = &(!&model.camera.transformation())
                            .unwrap()
                            * &ctx.mouse_position.as_f32();

                        // model.cursor_pos = pt2(
                        //     mouse_position.x as f32 / window_scale_factor as f32 + window_rect.x.start,
                        //     -mouse_position.y as f32 / window_scale_factor as f32 - window_rect.y.start,
                        // );

                        if !model.old_stuff.e0.update(ctx.mouse_position_in_world_space) {
                            model.old_stuff.e1.update(ctx.mouse_position_in_world_space);
                        }
                    }
                    MousePressed(button) => {
                        return;

                        if model.old_stuff.e0.ellipse.eq()(
                            *model
                                .event_handler_context
                                .mouse_position_in_world_space
                                .x() as f32,
                            *model
                                .event_handler_context
                                .mouse_position_in_world_space
                                .y() as f32,
                        ) < 0.
                        {
                            match button {
                                MouseButton::Left => model.old_stuff.e0.is_grabbed_to_move = true,
                                MouseButton::Right => {
                                    model.old_stuff.e0.is_grabbed_to_rotate = true
                                }
                                MouseButton::Middle => {
                                    model.old_stuff.e0.is_grabbed_to_scale = true
                                }
                                _ => {}
                            }
                        }
                        if model.old_stuff.e1.ellipse.eq()(
                            *ctx.mouse_position_in_world_space.x() as f32,
                            *ctx.mouse_position_in_world_space.y() as f32,
                        ) < 0.
                        {
                            match button {
                                MouseButton::Left => model.old_stuff.e1.is_grabbed_to_move = true,
                                MouseButton::Right => {
                                    model.old_stuff.e1.is_grabbed_to_rotate = true
                                }
                                MouseButton::Middle => {
                                    model.old_stuff.e1.is_grabbed_to_scale = true
                                }
                                _ => {}
                            }
                        }
                    }
                    MouseReleased(mouse_button) => {
                        model.old_stuff.e0.is_grabbed_to_move = false;
                        model.old_stuff.e0.is_grabbed_to_rotate = false;
                        model.old_stuff.e0.is_grabbed_to_scale = false;
                        model.old_stuff.e1.is_grabbed_to_move = false;
                        model.old_stuff.e1.is_grabbed_to_rotate = false;
                        model.old_stuff.e1.is_grabbed_to_scale = false;
                    }
                    MouseEntered => {}
                    MouseExited => {}
                    MouseWheel(mouse_scroll_delta, touch_phase) => {
                        let delta_to_y = |a: MouseScrollDelta| -> f32 {
                            match a {
                                MouseScrollDelta::LineDelta(_, y) => y,
                                MouseScrollDelta::PixelDelta(physical_position) => {
                                    physical_position.y as f32 / 10.
                                }
                            }
                        };

                        let angle_delta_to_scale_division = |angle_delta: f32| {
                            let base: f32 = 1.2;

                            base.powf(angle_delta)
                        };

                        let angle_delta_to_translation_delta = |angle_delta: f32| {
                            let velocity: f32 = 10.; // px per step
                            return velocity * angle_delta;
                        };

                        let position = ctx.mouse_position.as_f32();
                        let y = delta_to_y(mouse_scroll_delta);

                        if ctx.control {
                            // zoom
                            if ctx.center_on_vessel_mode {
                                let target_point = model
                                    .old_stuff
                                    .e1
                                    .ellipse
                                    .point_on_ellipse(model.vessel_orbit.anomaly);

                                model.camera.concat_scale_centered(
                                    angle_delta_to_scale_division(y),
                                    target_point,
                                    target_point,
                                );

                                return; // to prevent setting `center_on_vessel_mode` to false
                            } else {
                                model.camera.concat_scale_centered(
                                    angle_delta_to_scale_division(y),
                                    position,
                                    position,
                                );
                            }
                        } else if ctx.shift {
                            // scroll horizontally
                            model
                                .camera
                                .add_translation((angle_delta_to_translation_delta(y), 0.).into());
                        } else {
                            // scroll vertically
                            model
                                .camera
                                .add_translation((0., angle_delta_to_translation_delta(y)).into());
                        }

                        ctx.center_on_vessel_mode = false
                    }

                    Resized { .. } => {}
                    HoveredFile(path_buf) => todo!(),
                    DroppedFile(path_buf) => todo!(),
                    HoveredFileCancelled => todo!(),
                    Touch(touch_event) => todo!(),
                    TouchPressure(touchpad_pressure) => todo!(),
                    Focused => {}
                    Unfocused => {}
                    Closed => {}
                },
                None => {}
            }
        }
        Event::DeviceEvent(device_id, device_event) => {}
        Event::Update(update) => {}
        Event::Suspended => println!("Suspended: {:?}", event),
        Event::Resumed => println!("Resumed: {:?}", event),
    }
}
