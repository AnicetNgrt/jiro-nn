use super::{model::Model, Data};

pub trait Optimizer<'opgraph, T: Model + Data<'opgraph>>: Model {
    fn get_param(&self) -> &T;
    fn step(&mut self, incoming_grad: &T);
}

pub trait OptimizerBuilder<'opgraph, T: Model + Data<'opgraph>> {
    fn build(&self, parameter: T) -> Box<dyn Optimizer<'opgraph, T> + 'opgraph>;
    fn clone_box(&self) -> Box<dyn OptimizerBuilder<'opgraph, T> + 'opgraph>;
}
