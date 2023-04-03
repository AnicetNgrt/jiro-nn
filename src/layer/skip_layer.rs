use nalgebra::SVector;

use crate::layer::Layer;

pub struct SkipLayer<const I: usize>;

impl<const I: usize> Layer<I, I> for SkipLayer<I> {
    fn forward(&mut self, input: nalgebra::SVector<f64, I>) -> SVector<f64, I> {
        input
    }

    fn backward(
        &mut self,
        output_gradient: nalgebra::SVector<f64, I>,
        _learning_rate: f64,
    ) -> SVector<f64, I> {
        output_gradient
    }
}
