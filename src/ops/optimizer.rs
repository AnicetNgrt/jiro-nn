use super::{model::Model, Data};

pub trait Optimizer<'g, T: Model + Data<'g>>: Model {
    fn get_param(&self) -> &T;
    fn step(&mut self, incoming_grad: &T);
}

pub trait OptimizerBuilder<'g, T: Model + Data<'g>> {
    fn build(&self, parameter: T) -> Box<dyn Optimizer<'g, T> + 'g>;
    fn clone_box(&self) -> Box<dyn OptimizerBuilder<'g, T> + 'g>;
}
