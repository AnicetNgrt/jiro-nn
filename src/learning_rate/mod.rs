use serde::{Serialize, Deserialize};

use self::{inverse_time_decay::InverseTimeDecay, piecewise_constant::PiecewiseConstant};

pub mod inverse_time_decay;
pub mod piecewise_constant;

// good resource https://arxiv.org/pdf/1510.04609.pdf

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    Constant(f64),
    InverseTimeDecay(InverseTimeDecay),
    PiecewiseConstant(PiecewiseConstant),
}

impl LearningRateSchedule {
    pub fn get_learning_rate(&self, epoch: usize) -> f64 {
        match self {
            LearningRateSchedule::InverseTimeDecay(schedule) => schedule.get_learning_rate(epoch),
            LearningRateSchedule::PiecewiseConstant(schedule) => schedule.get_learning_rate(epoch),
            LearningRateSchedule::Constant(c) => *c,
        }
    }
}