use crate::linalg::{Matrix, MatrixTrait, Scalar};

pub trait Model {
    fn get_learnable_params_count(&self) -> usize;
    fn load_learnable_params(&mut self, params: Vec<Scalar>);
    fn get_learnable_params(&self) -> Vec<Scalar>;
}

impl Model for Scalar {
    fn get_learnable_params_count(&self) -> usize {
        1
    }

    fn load_learnable_params(&mut self, params: Vec<Scalar>) {
        *self = *params
            .get(0)
            .expect("Scalar params should have one parameter.");
    }

    fn get_learnable_params(&self) -> Vec<Scalar> {
        vec![*self]
    }
}

impl Model for Matrix {
    fn get_learnable_params_count(&self) -> usize {
        let dims = self.dim();
        dims.0 * dims.1
    }

    fn load_learnable_params(&mut self, params: Vec<Scalar>) {
        let ncols = params
            .first()
            .expect("First element should be the number of columns of the parameter matrix.");
        let ncols = *ncols as usize;
        let nrows = params.len() / ncols;
        let parameter = Self::from_iter(nrows, ncols, params.into_iter().skip(1));
        *self = parameter;
    }

    fn get_learnable_params(&self) -> Vec<Scalar> {
        let (_, ncols) = self.dim();
        let mut params = vec![ncols as Scalar];
        let data = self.get_data_col_leading();
        for col in 0..ncols {
            params.extend(data[col].iter().cloned());
        }
        params
    }
}

macro_rules! impl_model_no_params {
    () => {
        fn get_learnable_params_count(&self) -> usize {
            0
        }

        fn load_learnable_params(&mut self, _params: Vec<Scalar>) {}

        fn get_learnable_params(&self) -> Vec<Scalar> {
            vec![]
        }
    };
}

pub(crate) use impl_model_no_params;

macro_rules! impl_model_from_model_fields {
    ($p:ident) => {
        fn get_learnable_params_count(&self) -> usize {
            self.$p.get_learnable_params_count()
        }

        fn load_learnable_params(&mut self, params: Vec<Scalar>) {
            self.$p.load_learnable_params(params);
        }

        fn get_learnable_params(&self) -> Vec<Scalar> {
            self.$p.get_learnable_params()
        }
    };

    ($p1:ident, $p2:ident) => {
        fn get_learnable_params_count(&self) -> usize {
            self.$p1.get_learnable_params_count() + self.$p2.get_learnable_params_count()
        }

        fn load_learnable_params(&mut self, params: Vec<Scalar>) {
            let params1_count = self.$p1.get_learnable_params_count();
            let (params1_params, params2_params) = params.split_at(params1_count);
            self.$p1.load_learnable_params(params1_params.to_vec());
            self.$p2.load_learnable_params(params2_params.to_vec());
        }

        fn get_learnable_params(&self) -> Vec<Scalar> {
            let mut params = self.$p1.get_learnable_params();
            params.extend(self.$p2.get_learnable_params());
            params
        }
    };
}

pub(crate) use impl_model_from_model_fields;
