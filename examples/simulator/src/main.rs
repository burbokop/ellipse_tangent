use std::ops::Range;

mod md_array;
mod plot;

use chromosome::{Chromosome, Fitness, FitnessSelector, SimulationIter};
use ellipse_tangent::{
    ellipse::{Ellipse, TangentDirection},
    line::Line,
    utils::deg_to_rot,
};
use nannou::{
    draw::{primitive, Drawing},
    image::{DynamicImage, GenericImage as _, GenericImageView, RgbaImage},
    prelude::*,
};
use nannou_egui::{self, egui, Egui};

use crate::md_array::MdArray;

#[derive(Debug)]
struct TangentFitness {
    ellipse0: Ellipse,
    ellipse1: Ellipse,
    max_err: f32,
}

#[derive(Debug)]
struct TangentSegmentFitness {
    ellipse0: Ellipse,
    ellipse1: Ellipse,
    max_err: f32,
}

impl Fitness for TangentFitness {
    type Value = f32;

    fn fitness(&self, chromosome: &Chromosome<Self::Value>) -> Self::Value {
        let line = Line {
            k: chromosome.genes[0].tan(),
            d: chromosome.genes[1],
        };
        (self.ellipse0.intersection_discriminant(line).abs()
            + self.ellipse1.intersection_discriminant(line).abs())
        .abs()
    }

    fn is_ideal_fitness(&self, fitness: Self::Value) -> bool {
        fitness.abs() < self.max_err
    }
}

impl Fitness for TangentSegmentFitness {
    type Value = f32;

    fn fitness(&self, chromosome: &Chromosome<Self::Value>) -> Self::Value {
        match Line::from_points(
            chromosome.genes[0],
            chromosome.genes[1],
            chromosome.genes[2],
            chromosome.genes[3],
        ) {
            Some(line) => (self.ellipse0.intersection_discriminant(line).abs()
                + self.ellipse1.intersection_discriminant(line).abs()
                + self.ellipse0.eq()(chromosome.genes[0], chromosome.genes[1]).abs()
                + self.ellipse1.eq()(chromosome.genes[2], chromosome.genes[3]).abs())
            .abs(),
            None => Self::Value::MAX,
        }
    }

    fn is_ideal_fitness(&self, fitness: Self::Value) -> bool {
        fitness.abs() < self.max_err
    }
}

struct Settings {
    theta: f32,
    scale: f32,
}

struct EllipseState {
    ellipse: Ellipse,
    is_grabbed_to_move: bool,
    is_grabbed_to_rotate: bool,
    is_grabbed_to_scale: bool,
}

impl EllipseState {
    fn update(&mut self, cursor_pos: Point2) -> bool {
        if self.is_grabbed_to_move {
            self.ellipse.x = cursor_pos.x;
            self.ellipse.y = cursor_pos.y;
            true
        } else if self.is_grabbed_to_rotate {
            let rt = deg_to_rot(cursor_pos.x);
            self.ellipse.r = rt.0;
            self.ellipse.i = rt.1;
            true
        } else if self.is_grabbed_to_scale {
            self.ellipse.a = cursor_pos.x / 10.;
            self.ellipse.b = cursor_pos.y / 10.;
            true
        } else {
            false
        }
    }
}

struct Windows {
    main_window: WindowId,
    plot_window: WindowId,
}

struct Model<R: rand::RngCore> {
    e0: EllipseState,
    e1: EllipseState,
    common_tangents: Vec<(Line, TangentDirection)>,
    cursor_pos: Point2,
    sim: SimulationIter<f32, Range<f32>, FitnessSelector<TangentFitness>, R>,
    population: Vec<Chromosome<f32>>,
    settings: Settings,
    egui: Egui,
    image: DynamicImage,
    windows: Windows,
    plot_magnification: (f32, f32),
    plot_magnification_change_axis_y: bool,
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model<impl rand::RngCore> {
    let mut rng = rand::thread_rng();

    let main_window_id = app
        .new_window()
        .title("Genetic algorithm of finding common tangent to two ellipses")
        .size(1000, 480)
        .view(view::<rand::rngs::ThreadRng>)
        .raw_event(raw_window_event::<rand::rngs::ThreadRng>)
        .build()
        .unwrap();

    let plot_window_id = plot::new_plot_window(app);

    let window = app.window(main_window_id).unwrap();
    let egui = Egui::from_window(&window);

    let ellipse0 = Ellipse::new(100., 100., 40., 70., deg_to_rad(15.));
    let ellipse1 = Ellipse::new(-30., -100., 20., 80., deg_to_rad(300.));

    let initial_renge = -100.0..100.;
    let chromosome_size = 2;

    let initial_population = (0..8)
        .into_iter()
        .map(|_| Chromosome::<f32>::new_random(chromosome_size, initial_renge.clone(), &mut rng));

    let sim: SimulationIter<f32, Range<f32>, _, _> = SimulationIter::new(
        vec![0.01..0.2, 0.1..20.],
        0.1,
        initial_population.collect(),
        FitnessSelector::from(TangentFitness {
            ellipse0: ellipse0.clone(),
            ellipse1: ellipse1.clone(),
            max_err: 0.0005,
        }),
        rng,
    );

    //println!("sim: {:?}", sim);
    //println!("window.rect(): {:?}", window.rect());

    Model {
        e0: EllipseState {
            ellipse: ellipse0,
            is_grabbed_to_move: false,
            is_grabbed_to_rotate: false,
            is_grabbed_to_scale: false,
        },
        e1: EllipseState {
            ellipse: ellipse1,
            is_grabbed_to_move: false,
            is_grabbed_to_rotate: false,
            is_grabbed_to_scale: false,
        },
        common_tangents: vec![],
        cursor_pos: pt2(0., 0.),
        sim,
        population: vec![],
        settings: Settings {
            theta: 0.,
            scale: 1.,
        },
        egui,
        image: DynamicImage::ImageRgba8(RgbaImage::new(
            window.rect().w() as u32,
            window.rect().h() as u32,
        )),
        windows: Windows {
            main_window: main_window_id,
            plot_window: plot_window_id,
        },
        plot_magnification: (1., 1.),
        plot_magnification_change_axis_y: false,
    }
}

fn fill_image<R: rand::RngCore>(app: &App, model: &mut Model<R>) {
    let window_rect = app.window(model.windows.main_window).unwrap().rect();
    //println!("window_rect: {:?}, {}", window_rect, window_rect.x.start);

    let _pt_to_img = |pt: Point2| {
        (
            (pt.x - window_rect.x.start) as usize,
            (pt.y - window_rect.y.start) as usize,
        )
    };
    let pt_from_img = |x: usize, y: usize| {
        pt2(
            x as f32 + window_rect.x.start,
            -(y as f32) - window_rect.y.start,
        )
    };

    let mut array: MdArray<f32, 2> = MdArray::new(
        0.,
        model.image.width() as usize,
        model.image.height() as usize,
    );

    let k = deg_to_rad(model.settings.theta).tan();
    for y in (0..array.height()).step_by(4) {
        for x in (0..array.width()).step_by(4) {
            let pt = pt_from_img(x, y);

            //*array.at_mut(x, y) = model.e0.eq()(pt.x, pt.y);

            *array.at_mut(x, y) = model.e0.ellipse.intersection_discriminant(Line {
                k,
                d: pt.y - k * pt.x,
            }) * model.e1.ellipse.intersection_discriminant(Line {
                k,
                d: pt.y - k * pt.x,
            })
        }
    }

    //println!("BEFORE: {:?}", &array.raw()[0..256]);
    let array = array.partially_normalized();
    //println!("AFTER: {:?}", &array.raw()[0..256]);

    //let mut image = RgbaImage::new(window_rect.w() as u32, window_rect.h() as u32);
    for y in 0..array.height() {
        for x in 0..array.width() {
            let v = *array.at(x, y);
            let v_abs = (v.abs() * 255. * model.settings.scale) as u8;
            model.image.put_pixel(
                x as u32,
                y as u32,
                nannou::image::Rgba(if v < 0. {
                    [255, 0, 0, v_abs]
                } else {
                    [0, 255, 0, v_abs]
                }),
            );

            //if  > 0
            //let gray = (array.at(x, y) * 255.) as u8;
            //
            //let argb = (array.at(x, y) * (0x00ffffff as f32)) as u32;
            //let a = (argb << 24) as u8;
            //let r = (argb << 16) as u8;
            //let g = (argb << 8) as u8;
            //let b = (argb << 0) as u8;
            //
            ////*image.get_pixel_mut(x as u32, y as u32) = nannou::image::Rgba([r,g,b,127]);
            //*image.get_pixel_mut(x as u32, y as u32) = nannou::image::Rgba([gray,0,0,127]);
        }
    }
}

fn update<R: rand::RngCore>(app: &App, model: &mut Model<R>, update: Update) {
    if let Some(population) = model.sim.next() {
        model.population = population;
        //println!("population: {:?}", model.population);
    } else {
        //println!("nothing to do");
    }

    let mut acc = 0_f32;

    {
        let egui = &mut model.egui;
        let settings = &mut model.settings;
        egui.set_elapsed_time(update.since_start);
        let ctx = egui.begin_frame();

        let theta = &mut settings.theta;

        egui::Window::new("Settings").show(&ctx, |ui| {
            // Scale slider
            ui.label(format!("E0: {:?}", &model.e0.ellipse));
            ui.label(format!("E1: {:?}", &model.e1.ellipse));

            ui.label("Scale:");
            ui.add(egui::Slider::new(&mut settings.scale, 0.1..=10000.));
            ui.label("θ:");
            ui.add(egui::Slider::new(theta, (0.)..=360.).step_by(1.));
            ui.label("acc:");
            ui.add(egui::Slider::new(&mut acc, (0.01)..=1.).step_by(0.01));

            let k = deg_to_rad(*theta).tan();
            ui.label(format!("K: {}", k));

            ui.label(format!("ell_0.D: {:?}", model.e0.ellipse.tangent_d(k)));
            ui.label(format!("ell_1.D: {:?}", model.e1.ellipse.tangent_d(k)));

            let outer = model.e0.ellipse.outer_tangents_sdf(&model.e1.ellipse, k);
            ui.label(format!("result: {:?}", outer));

            let i_d = model
                .e0
                .ellipse
                .common_tangents_intermediate_data(&model.e1.ellipse);
            ui.label(format!("i.d.: {:#?}", i_d));
        });
    }

    model.common_tangents = model.e0.ellipse.common_tangents3(&model.e1.ellipse);

    fill_image(app, model);
}

fn draw_line_by_kd(draw: &Draw, k: f32, d: f32) -> Drawing<primitive::Line> {
    let start = pt2(-400., -400.);
    let end = pt2(400., 400.);

    let x0 = start.x;
    let x1 = end.x;

    let y0 = k * x0 + d;
    let y1 = k * x1 + d;

    draw.line().points(pt2(x0, y0), pt2(x1, y1))
}

fn draw_line(draw: &Draw, line: Line) -> Drawing<primitive::Line> {
    draw_line_by_kd(draw, line.k, line.d)
}

fn raw_window_event<R: rand::RngCore>(
    app: &App,
    model: &mut Model<R>,
    event: &nannou::winit::event::WindowEvent,
) {
    if let Some(window) = app.window(model.windows.main_window) {
        let window_rect = window.rect();
        let window_scale_factor = window.scale_factor();

        match event {
            nannou::winit::event::WindowEvent::Resized { .. } => {
                model.image = DynamicImage::ImageRgba8(RgbaImage::new(
                    window.rect().w() as u32,
                    window.rect().h() as u32,
                ))
            }
            nannou::winit::event::WindowEvent::CursorMoved {
                device_id,
                position,
                modifiers,
            } => {
                model.cursor_pos = pt2(
                    position.x as f32 / window_scale_factor as f32 + window_rect.x.start,
                    -position.y as f32 / window_scale_factor as f32 - window_rect.y.start,
                );

                if !model.e0.update(model.cursor_pos) {
                    model.e1.update(model.cursor_pos);
                }
            }
            nannou::winit::event::WindowEvent::MouseInput {
                device_id,
                state,
                button,
                modifiers,
            } => match state {
                nannou::event::ElementState::Pressed => {
                    if model.e0.ellipse.eq()(model.cursor_pos.x, model.cursor_pos.y) < 0. {
                        match button {
                            MouseButton::Left => model.e0.is_grabbed_to_move = true,
                            MouseButton::Right => model.e0.is_grabbed_to_rotate = true,
                            MouseButton::Middle => model.e0.is_grabbed_to_scale = true,
                            _ => {}
                        }
                    }
                    if model.e1.ellipse.eq()(model.cursor_pos.x, model.cursor_pos.y) < 0. {
                        match button {
                            MouseButton::Left => model.e1.is_grabbed_to_move = true,
                            MouseButton::Right => model.e1.is_grabbed_to_rotate = true,
                            MouseButton::Middle => model.e1.is_grabbed_to_scale = true,
                            _ => {}
                        }
                    }
                }
                nannou::event::ElementState::Released => {
                    model.e0.is_grabbed_to_move = false;
                    model.e0.is_grabbed_to_rotate = false;
                    model.e0.is_grabbed_to_scale = false;
                    model.e1.is_grabbed_to_move = false;
                    model.e1.is_grabbed_to_rotate = false;
                    model.e1.is_grabbed_to_scale = false;
                }
            },
            nannou::winit::event::WindowEvent::Touch(_) => todo!(),
            _ => {}
        }
        // Let egui handle things like keyboard and mouse input.
        model.egui.handle_raw_event(event);
    }
}

fn draw_ellipse(draw: &Draw, ellipse: &Ellipse) {
    draw.ellipse()
        .x(ellipse.x)
        .y(ellipse.y)
        .w(ellipse.a * 2.)
        .h(ellipse.b * 2.)
        .color(WHITE)
        .rotate(-f32::atan2(ellipse.i, ellipse.r));

    //for y in (ellipse.y - ellipse.b * 4.) as i32..(ellipse.y + ellipse.b * 4.) as i32 {
    //    for x in (ellipse.x - ellipse.a * 16.) as i32..(ellipse.x + ellipse.a * 16.) as i32 {
    //        let x = x as f32;
    //        let y = y as f32;
    //        if ellipse.eq()(x, y).abs() < 0.04 {
    //            draw.ellipse().color(DARKCYAN).radius(1.).x(x).y(y);
    //        }
    //    }
    //}
}

fn view<R: rand::RngCore>(app: &App, model: &Model<R>, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);

    draw_ellipse(&draw, &model.e0.ellipse);
    draw_ellipse(&draw, &model.e1.ellipse);

    let start = pt2(-400., -400.);
    let end = pt2(400., 400.);

    //for chromosome in &model.population {
    //    let p0 = pt2(chromosome.genes[0], chromosome.genes[1]);
    //    let p1 = pt2(chromosome.genes[2], chromosome.genes[3]);
    //    draw.line().points(p0, p1).color(GREENYELLOW).stroke_weight(2.);
    //}

    for chromosome in &model.population {
        let k = chromosome.genes[0].tan();
        let d = chromosome.genes[1];
        let p0 = pt2(start.x, k * start.x + d);
        let p1 = pt2(end.x, k * end.x + d);
        draw.line().points(p0, p1).color(PINK).stroke_weight(2.);
    }

    let texture = wgpu::Texture::from_image(app, &model.image);

    draw.texture(&texture);

    let k = deg_to_rad(model.settings.theta).tan();

    let e0d = model.e0.ellipse.tangent_d(k);
    let e1d = model.e1.ellipse.tangent_d(k);

    //println!("e0: {:?}", &model.e0.ellipse);
    //println!("e1: {:?}", &model.e1.ellipse);

    //println!("e0d: {:?}", e0d);
    //println!("e1d: {:?}", e1d);

    draw_line_by_kd(&draw, k, e0d.0)
        .stroke_weight(1.)
        .color(GREEN);
    draw_line_by_kd(&draw, k, e0d.1)
        .stroke_weight(1.)
        .color(LIGHTGREEN);
    draw_line_by_kd(&draw, k, e1d.0)
        .stroke_weight(1.)
        .color(BLUE);
    draw_line_by_kd(&draw, k, e1d.1)
        .stroke_weight(1.)
        .color(LIGHTBLUE);

    draw.ellipse()
        .stroke(VIOLET)
        .stroke_weight(2.)
        .radius(5.)
        .x(model.cursor_pos.x)
        .y(model.cursor_pos.y)
        .color(BLACK);

    for t in &model.common_tangents {
        draw_line(&draw, t.0).stroke_weight(3.).color(VIOLET);
        draw_line(&draw, t.0).stroke_weight(1.).color(match t.1 {
            TangentDirection::Left => BLACK,
            TangentDirection::Right => WHITE,
        });
    }

    {
        let mut prev_x = 0.;
        let mut prev_y = [0.; 2];
        let mut has_prev: bool = false;
        for i in -400..400 {
            let x = i as f32;
            let y = model.e0.ellipse.y(x);

            if has_prev {
                //println!("{}, {:?} -> {}, {:?}", prev_x, prev_y, x, y);
                if !y[0].is_nan() && !prev_y[0].is_nan() {
                    draw.line()
                    .points(
                        pt2(prev_x, prev_y[0]),
                        pt2(x, y[0]),
                    )
                    .stroke_weight(3.)
                    .color(YELLOW);
                }
                if !y[1].is_nan() && !prev_y[1].is_nan() {
                    draw.line()
                    .points(
                        pt2(prev_x, prev_y[1]),
                        pt2(x, y[1]),
                    )
                    .stroke_weight(3.)
                    .color(GREEN);
                }
            }

            has_prev = true;
            prev_x = x;
            prev_y = y;
        }
    }

    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}
