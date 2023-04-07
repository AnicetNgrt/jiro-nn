use nalgebra::{DVector, DMatrix};

use crate::layer::Layer;

pub struct DenseLayer {
    // i inputs, j outputs, i x j connections
    input: Option<DVector<f64>>,
    // j x i connection weights
    weights: DMatrix<f64>,
    // j output biases
    biases: DVector<f64>,
}

impl DenseLayer {
    pub fn new(i: usize, j: usize) -> Self {
        let weights = DMatrix::<f64>::new_random(j, i);
        let biases = DVector::<f64>::new_random(j);

        Self {
            weights: weights.add_scalar(-0.5) * 2.,
            biases: biases.add_scalar(-0.5) * 2.,
            input: None,
        }
    }
}

impl Layer for DenseLayer {
    fn forward(&mut self, input: DVector<f64>) -> DVector<f64> {
        // Y = W . X + B
        let res = &self.weights * &input + &self.biases;
        self.input = Some(input);
        res
    }

    fn backward(
        &mut self,
        output_gradient: DVector<f64>,
        learning_rate: f64,
    ) -> DVector<f64> {
        let input = self.input.clone().unwrap();
        // [f(g(x))]' = f'(g(x)) × g'(x)

        // ∂E/∂W = ∂E/∂Y . ∂Y/∂W
        // But ∂yj/∂wji != 0 and ∂yj/∂wki == 0, k != j
        // => ∂E/∂wji = ∂E/∂yj . ∂yj/∂wji
        // And ∂yj/∂wji = xi
        // => ∂E/∂wji = ∂E/∂yj . xi
        // => [ ∂E/∂y1, ∂E/∂y2, ... ∂E/∂yj ]^t . [ x1, x2, ... xi ]
        // => ∂E/∂W = ∂E/∂Y . X^t
        let weights_gradient = &output_gradient * &input.transpose();

        // ∂E/∂B = ∂E/∂Y . ∂Y/∂B
        // But ∂yj/∂bj != 0 and ∂yj/∂bk == 0, k != j
        // => ∂E/∂bj = ∂E/∂yj
        // => ∂E/∂B = ∂E/∂Y
        let biases_gradient = &output_gradient;

        // ∂E/∂X = ∂E/∂Y . ∂Y/∂X
        // With ∂E/∂xi = ∂E/∂y1 ∂y1/∂xi + ∂E/∂y2 ∂y2/∂xi + ... + ∂E/∂yj ∂yj/∂xi
        // But yj = ... + xi*wji + ...  so  ∂yj/∂xi = wji
        // => ∂E/∂xi = ∂E/∂y1 w1i + ∂E/∂y2 w2i + ... + ∂E/∂yj wji
        // => ∂E/∂X = W^t . [∂E/∂y1, ∂E/∂y2, ... ∂E/∂yj]
        // => ∂E/∂X = W^t . ∂E/∂Y
        let input_gradient = self.weights.transpose() * &output_gradient;

        // (Stochastic) Gradient descent -> Following the negative gradient (computed on discrete xs) & converging to the minimum
        self.weights = &self.weights - (learning_rate * weights_gradient);
        self.biases = &self.biases - (learning_rate * biases_gradient);

        input_gradient
    }
}
