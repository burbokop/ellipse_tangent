use crate::Model;
use burbomath::Point;
use nannou::event;
use nannou::App;
use nannou::{
    image::{DynamicImage, RgbaImage},
    prelude::*,
    text::Font,
};


pub struct EventContext {
    control: bool,
    shift: bool,
    mouse_position: Point<i32>,
    mouse_position_in_world_space: Point<f32>,
}

impl Default for EventContext {
    fn default() -> Self {
        Self {
            control: Default::default(),
            shift: Default::default(),
            mouse_position: (0, 0).into(),
            mouse_position_in_world_space: (0., 0.).into(),
        }
    }
}

pub fn event<R: rand::RngCore>(app: &App, model: &mut Model<R>, event: Event) {
    match event {
        Event::WindowEvent { id, simple } => {
            let w = app.window(id).unwrap();

            match simple {
                Some(e) => match e {
                    Moved(vec2) => todo!(),
                    KeyPressed(event::Key::LShift | event::Key::RShift) => {
                        model.event_context.shift = true
                    }
                    KeyReleased(event::Key::LShift | event::Key::RShift) => {
                        model.event_context.shift = false
                    }
                    KeyPressed(event::Key::LControl | event::Key::RControl) => {
                        model.event_context.control = true
                    }
                    KeyReleased(event::Key::LControl | event::Key::RControl) => {
                        model.event_context.control = false
                    }
                    KeyPressed(..) => {}
                    KeyReleased(..) => {}
                    ReceivedCharacter(_) => todo!(),
                    MouseMoved(vec2) => {
                        model.event_context.mouse_position = (vec2.x as i32, vec2.y as i32).into();
                        model.event_context.mouse_position_in_world_space =
                            &(!&model.camera.transformation()).unwrap()
                                * &model.event_context.mouse_position.as_f32();

                        // model.cursor_pos = pt2(
                        //     mouse_position.x as f32 / window_scale_factor as f32 + window_rect.x.start,
                        //     -mouse_position.y as f32 / window_scale_factor as f32 - window_rect.y.start,
                        // );

                        if !model
                            .e0
                            .update(model.event_context.mouse_position_in_world_space)
                        {
                            model
                                .e1
                                .update(model.event_context.mouse_position_in_world_space);
                        }
                    }
                    MousePressed(button) => {
                        return;

                        if model.e0.ellipse.eq()(
                            *model.event_context.mouse_position_in_world_space.x() as f32,
                            *model.event_context.mouse_position_in_world_space.y() as f32,
                        ) < 0.
                        {
                            match button {
                                MouseButton::Left => model.e0.is_grabbed_to_move = true,
                                MouseButton::Right => model.e0.is_grabbed_to_rotate = true,
                                MouseButton::Middle => model.e0.is_grabbed_to_scale = true,
                                _ => {}
                            }
                        }
                        if model.e1.ellipse.eq()(
                            *model.event_context.mouse_position_in_world_space.x() as f32,
                            *model.event_context.mouse_position_in_world_space.y() as f32,
                        ) < 0.
                        {
                            match button {
                                MouseButton::Left => model.e1.is_grabbed_to_move = true,
                                MouseButton::Right => model.e1.is_grabbed_to_rotate = true,
                                MouseButton::Middle => model.e1.is_grabbed_to_scale = true,
                                _ => {}
                            }
                        }
                    }
                    MouseReleased(mouse_button) => {
                        model.e0.is_grabbed_to_move = false;
                        model.e0.is_grabbed_to_rotate = false;
                        model.e0.is_grabbed_to_scale = false;
                        model.e1.is_grabbed_to_move = false;
                        model.e1.is_grabbed_to_rotate = false;
                        model.e1.is_grabbed_to_scale = false;
                    }
                    MouseEntered => {}
                    MouseExited => {}
                    MouseWheel(mouse_scroll_delta, touch_phase) => {
                        let delta_to_y = |a: MouseScrollDelta| -> f32 {
                            match a {
                                MouseScrollDelta::LineDelta(_, y) => y,
                                MouseScrollDelta::PixelDelta(physical_position) => todo!(),
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

                        let position = model.event_context.mouse_position.as_f32();
                        let y = delta_to_y(mouse_scroll_delta);

                        if model.event_context.control {
                            // zoom
                            model.camera.concat_scale_centered(
                                angle_delta_to_scale_division(y),
                                position,
                                position,
                            );
                        } else if model.event_context.shift {
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
