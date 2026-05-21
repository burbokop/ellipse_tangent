use burbomath::Angle;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ellipse_tangent::ellipse::Ellipse;

fn common_tangents(c: &mut Criterion) {
    let ellipse0 = Ellipse::from_angle(
        (100., 100.).into(),
        (40., 70.).into(),
        Angle::from_degrees(15.),
    );
    let ellipse1 = Ellipse::from_angle(
        (-30., -100.).into(),
        (20., 80.).into(),
        Angle::from_degrees(300.),
    );

    c.bench_function("common_tangents", |b| {
        b.iter(|| ellipse0.common_tangents(black_box(&ellipse1)))
    });
}

criterion_group!(benches, common_tangents);
criterion_main!(benches);
