use super::{model::Model, Data};

pub trait Optimizer<T: Model + Data>: Model {
    fn get_param(&self) -> &T;
    fn step(&mut self, incoming_grad: &T);
}