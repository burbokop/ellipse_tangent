use std::ops::Range;

mod ellipse;
mod line;
use chromosome::{Chromosome, Fitness, FitnessSelector, SimulationIter};
use ellipse::Ellipse;
use line::Line;
use nannou::prelude::*;

fn deg_to_rad(deg: f32) -> f32 {
    deg * std::f32::consts::PI / 180.
}

#[derive(Debug)]
struct TangentFitness {
    ellipse0: Ellipse,
    ellipse1: Ellipse,
    max_err: f32
}

#[derive(Debug)]
struct TangentSegmentFitness {
    ellipse0: Ellipse,
    ellipse1: Ellipse,
    max_err: f32
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

struct Model<R: rand::RngCore> {
    e0: Ellipse,
    e1: Ellipse,
    sim: SimulationIter<f32, Range<f32>, FitnessSelector<TangentFitness>, R>,
    population: Vec<Chromosome<f32>>,
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model<impl rand::RngCore> {
    let mut rng = rand::thread_rng();

    let _w_id = app
        .new_window()
        .title("Genetic algorithm of finding common tangent to two ellipses")
        .size(1000, 480)
        .view(view::<rand::rngs::ThreadRng>)
        .build()
        .unwrap();

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

    println!("sim: {:?}", sim);

    Model {
        e0: ellipse0,
        e1: ellipse1,
        sim,
        population: vec![],
    }
}

fn update<R: rand::RngCore>(_: &App, model: &mut Model<R>, _update: Update) {
    if let Some(population) = model.sim.next() {
        model.population = population;
        println!("population: {:?}", model.population);
    } else {
        println!("nothing to do");
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

    for y in (ellipse.y - ellipse.b * 4.) as i32..(ellipse.y + ellipse.b * 4.) as i32 {
        for x in (ellipse.x - ellipse.a * 16.) as i32..(ellipse.x + ellipse.a * 16.) as i32 {
            let x = x as f32;
            let y = y as f32;

            if ellipse.eq()(x, y).abs() < 0.04 {
                draw.ellipse().color(DARKCYAN).radius(1.).x(x).y(y);
            }
        }
    }
}

fn view<R: rand::RngCore>(app: &App, model: &Model<R>, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);

    draw_ellipse(&draw, &model.e0);
    draw_ellipse(&draw, &model.e1);

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

    draw.to_frame(app, &frame).unwrap();
}
