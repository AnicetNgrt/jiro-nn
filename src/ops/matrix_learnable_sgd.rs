use crate::{linalg::{Matrix, MatrixTrait, Scalar}, learning_rate::LearningRateScheduler};

use super::{Op, matrix_learnable::MatrixLearnable};

pub struct MatrixLearnableSGD {
    parameter: Matrix,
    learning_rate: Box<dyn LearningRateScheduler>,
}

impl Op<Matrix, Matrix> for MatrixLearnableSGD {
    fn forward_inference(&self, input: &Matrix) -> Matrix {
        input.clone()
    }

    fn forward(&mut self, input: &Matrix) -> Matrix {
        input.clone()
    }

    fn backward(&mut self, incoming_grad: &Matrix) -> Matrix {
        let lr = self.learning_rate.get_learning_rate();
        self.parameter = self.parameter.component_sub(&incoming_grad.scalar_mul(lr));
        self.learning_rate.increment_step();
        incoming_grad.clone()
    }

    fn get_learnable_params_count(&self) -> usize {
        let dims = self.parameter.dim();
        dims.0 * dims.1
    }

    fn load_learnable_params(&mut self, params: Vec<Scalar>) {
        let ncols = params.first().expect("First element should be the number of columns of the parameter matrix.");
        let ncols = *ncols as usize;
        let nrows = params.len() / ncols;
        let parameter = Matrix::from_iter(nrows, ncols, params.into_iter().skip(1));
        self.parameter = parameter;
    }

    fn get_learnable_params(&self) -> Vec<Scalar> {
        let (_, ncols) = self.parameter.dim();
        let mut params = vec![ncols as Scalar];
        let data = self.parameter.get_data_col_leading();
        for col in 0..ncols {
            params.extend(data[col].iter().cloned());
        }
        params
    }
}

impl MatrixLearnable for MatrixLearnableSGD {
    fn get(&self) -> &Matrix {
        &self.parameter
    }
}