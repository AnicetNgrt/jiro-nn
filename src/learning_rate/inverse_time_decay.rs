use serde::{Deserialize, Serialize};

use crate::linalg::Scalar;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InverseTimeDecay {
    pub initial_learning_rate: Scalar,
    pub decay_steps: Scalar,
    pub decay_rate: Scalar,
    #[serde(default)]
    pub staircase: bool,
}

impl InverseTimeDecay {
    pub fn new(
        initial_learning_rate: Scalar,
        decay_steps: Scalar,
        decay_rate: Scalar,
        staircase: bool,
    ) -> Self {
        Self {
            initial_learning_rate,
            decay_steps,
            decay_rate,
            staircase,
        }
    }

    pub fn get_learning_rate(&self, epoch: usize) -> Scalar {
        let mut learning_rate = self.initial_learning_rate
            / (1. + self.decay_rate * (epoch as Scalar / self.decay_steps));
        if self.staircase {
            learning_rate = self.initial_learning_rate
                / (1. + self.decay_rate * (epoch as Scalar / self.decay_steps).floor());
        }
        learning_rate
    }
}
