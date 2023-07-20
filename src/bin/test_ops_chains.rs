use jiro_nn::{
    linalg::{Matrix, MatrixTrait},
    ops::model_op_builder::{ModelBuilderOrigin, OpBuild},
};

fn main() {
    let model_builder = OpBuild::new_op_builder(ModelBuilderOrigin::<Matrix, Matrix>::new());
    model_builder
        .dense(18)
        .everything_sgd_optimized()
        .constant_learning_rate(0.1)
        .end()
        .end()
        .tanh()
        .custom_activation(
            |m| m.scalar_mul(3.0),
            |m| Matrix::constant(m.dim().0, m.dim().1, 3.0),
        );
}
