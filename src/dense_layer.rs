use nalgebra::{SMatrix, SVector};

use crate::layer::Layer;

// Translated from python https://www.youtube.com/watch?v=pauPCy_s0Ok
pub struct DenseLayer<const I: usize, const J: usize> {
    // i inputs, j outputs, i x j connections
    input: Option<SVector<f64, I>>,
    // j x i connection weights
    weights: SMatrix<f64, J, I>,
    // j output biases
    biases: SVector<f64, J>,
}

impl<const I: usize, const J: usize> DenseLayer<I, J> {
    pub fn new() -> Self {
        Self {
            weights: SMatrix::new_random(),
            biases: SVector::new_random(),
            input: None,
        }
    }
}

impl<const I: usize, const J: usize> Layer<I, J> for DenseLayer<I, J> {
    fn forward(&mut self, input: SVector<f64, I>) -> SVector<f64, J> {
        // Y = W . X + B
        self.input = Some(input);
        self.weights * input + self.biases
    }

    fn backward(
        &mut self,
        output_gradient: SVector<f64, J>,
        learning_rate: f64,
    ) -> SVector<f64, I> {
        let input = self.input.unwrap();

        // ∂E/∂W = ∂E/∂Y . ∂Y/∂W
        // But ∂yj/∂wji != 0 and ∂yj/∂wki == 0, k != j
        // => ∂E/∂wji = ∂E/∂yj . ∂yj/∂wji
        // And ∂yj/∂wji = xi
        // => ∂E/∂wji = ∂E/∂yj . xi
        // => [ ∂E/∂y1, ∂E/∂y2, ... ∂E/∂yj ]^t . [ x1, x2, ... xi ]
        // => ∂E/∂W = ∂E/∂Y . X^t
        let weights_gradient = output_gradient * input.transpose();

        // ∂E/∂B = ∂E/∂Y . ∂Y/∂B
        // But ∂yj/∂bj != 0 and ∂yj/∂bk == 0, k != j
        // => ∂E/∂bj = ∂E/∂yj
        // => ∂E/∂B = ∂E/∂Y
        let biases_gradient = output_gradient;

        // ∂E/∂X = ∂E/∂Y . ∂Y/∂X
        // With ∂E/∂xi = ∂E/∂y1 ∂y1/∂xi + ∂E/∂y2 ∂y2/∂xi + ... + ∂E/∂yj ∂yj/∂xi
        // But yj = ... + xi*wji + ...  so  ∂yj/∂xi = wji
        // => ∂E/∂xi = ∂E/∂y1 w1i + ∂E/∂y2 w2i + ... + ∂E/∂yj wji
        // => ∂E/∂X = W^t . [∂E/∂y1, ∂E/∂y2, ... ∂E/∂yj]
        // => ∂E/∂X = W^t . ∂E/∂Y
        let input_gradient = self.weights.transpose() * output_gradient;

        // Gradient descent -> Following the negative gradient & converging to the minimum
        self.weights = self.weights - learning_rate * weights_gradient;
        self.biases = self.biases - learning_rate * biases_gradient;

        input_gradient
    }
}
