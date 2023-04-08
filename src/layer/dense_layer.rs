use nalgebra::{DVector, DMatrix};

use crate::layer::Layer;

pub struct DenseLayer {
    // i inputs, j outputs, i x j connections
    input: Option<DMatrix<f64>>,
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
    fn forward(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        // Y = W . X + B
        let mut res = &input * &self.weights.transpose();
        // add the biases to each row of res
        res.column_iter_mut().zip(self.biases.iter()).for_each(|(mut col, bias)| {
            col.add_scalar_mut(*bias);
        });
        self.input = Some(input);
        // each row -> one batch of inputs for one neuron
        // each column -> inputs for each next neurons
        res
    }

    fn backward(
        &mut self,
        output_gradient: DMatrix<f64>,
        learning_rate: f64,
    ) -> DMatrix<f64> {
        let input = self.input.clone().unwrap();
        // [f(g(x))]' = f'(g(x)) × g'(x)

        //println!("outg: {:?} input:{:?}", output_gradient.shape(), input.shape());
        let weights_gradient = &output_gradient.transpose() * &input;

        let biases_gradient = &output_gradient.transpose().column_sum();

        //println!("weights: {:?} outg:{:?}", weights_gradient.transpose().shape(), biases_gradient.shape());
        let input_gradient = output_gradient * &self.weights;

        self.weights = &self.weights - (learning_rate * weights_gradient);
        self.biases = &self.biases - (learning_rate * biases_gradient);

        input_gradient
    }
}
