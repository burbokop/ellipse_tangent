use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ellipse_tangent::{ellipse::Ellipse, utils::deg_to_rad};

fn common_tangents(c: &mut Criterion) {
    let ellipse0 = Ellipse::new(100., 100., 40., 70., deg_to_rad(15.));
    let ellipse1 = Ellipse::new(-30., -100., 20., 80., deg_to_rad(300.));

    c.bench_function("common_tangents", |b| {
        b.iter(|| ellipse0.common_tangents(black_box(&ellipse1)))
    });
}

criterion_group!(benches, common_tangents);
criterion_main!(benches);
