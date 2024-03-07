use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ellipse_tangent::{ellipse::Ellipse, utils::deg_to_rad};
use rand::RngCore;
use rand::Rng as _;

fn new_rand_ellipse<R: RngCore>(rng: &mut R) -> Ellipse {
    let theta = rng.gen::<f32>();
    Ellipse {
        x: rng.gen::<f32>(),
        y: rng.gen::<f32>(),
        a: rng.gen::<f32>(),
        b: rng.gen::<f32>(),
        r: theta.cos(),
        i: theta.sin(),
    }
}

fn common_tangents(c: &mut Criterion) {
    // 1. HUAWEI CE0682 time:   [2.2347 µs 2.2374 µs 2.2409 µs]
    //    Intel i7      time:   [228.15 ns 228.46 ns 228.83 ns]
    // 2. HUAWEI CE0682 time:   [12.634 µs 12.640 µs 12.647 µs]
    //    Intel i7      time:   [1.2194 µs 1.2277 µs 1.2367 µs]
    c.bench_function("static common_tangents", |b| {
        let ellipse0 = Ellipse::new(100., 100., 40., 70., deg_to_rad(15.));
        let ellipse1 = Ellipse::new(-30., -100., 20., 80., deg_to_rad(300.));
        b.iter(|| ellipse0.common_tangents(black_box(&ellipse1)))
    });

    // 1. HUAWEI CE0682 time:   [1.9555 µs 2.1434 µs 2.3091 µs]
    //    Intel i7      time:   [178.22 ns 198.85 ns 218.76 ns]
    // 2. HUAWEI CE0682 time:   [10.869 µs 11.504 µs 12.108 µs]
    //    Intel i7      time:   [946.54 ns 1.0378 µs 1.1288 µs]
    let mut rng = rand::thread_rng();
    c.bench_function("random common_tangents", |b|{
        let ellipse0 = new_rand_ellipse(&mut rng);
        let ellipse1 = new_rand_ellipse(&mut rng);
        b.iter(|| ellipse0.common_tangents(black_box(&ellipse1)))
    });
}

criterion_group!(benches, common_tangents);
criterion_main!(benches);
