use serde::{Deserialize, Serialize};

use self::{inverse_time_decay::InverseTimeDecay, piecewise_constant::PiecewiseConstant};
use crate::linalg::Scalar;

pub mod inverse_time_decay;
pub mod piecewise_constant;

pub fn default_learning_rate() -> LearningRateSchedule {
    LearningRateSchedule::Constant(0.001)
}

// https://arxiv.org/pdf/1510.04609.pdf
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    Constant(Scalar),
    InverseTimeDecay(InverseTimeDecay),
    PiecewiseConstant(PiecewiseConstant),
}

impl LearningRateSchedule {
    pub fn get_learning_rate(&self, epoch: usize) -> Scalar {
        match self {
            LearningRateSchedule::InverseTimeDecay(schedule) => schedule.get_learning_rate(epoch),
            LearningRateSchedule::PiecewiseConstant(schedule) => schedule.get_learning_rate(epoch),
            LearningRateSchedule::Constant(c) => *c,
        }
    }
}

pub trait LearningRateScheduler {
    fn increment_step(&mut self);
    fn get_learning_rate(&self) -> Scalar;
}

pub struct ConstantLearningRate {
    learning_rate: Scalar,
}

impl ConstantLearningRate {
    pub fn new(learning_rate: Scalar) -> Self {
        Self { learning_rate }
    }
}

impl LearningRateScheduler for ConstantLearningRate {
    fn increment_step(&mut self) {}

    fn get_learning_rate(&self) -> Scalar {
        self.learning_rate
    }
}