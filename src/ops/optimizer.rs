use super::{model::Model, Data};

pub trait Optimizer<T: Model + Data>: Model {
    fn get_param(&self) -> &T;
    fn step(&mut self, incoming_grad: &T);
}

pub trait OptimizerBuilder<T: Model + Data> {
    fn build(&self, parameter: T) -> Box<dyn Optimizer<T>>;
    fn clone_box(&self) -> Box<dyn OptimizerBuilder<T>>;
}
