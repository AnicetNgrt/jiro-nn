use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InverseTimeDecay {
    pub initial_learning_rate: f64,
    pub decay_steps: f64,
    pub decay_rate: f64,
    pub staircase: bool,
}

impl InverseTimeDecay {
    pub fn new(
        initial_learning_rate: f64,
        decay_steps: f64,
        decay_rate: f64,
        staircase: bool,
    ) -> Self {
        Self {
            initial_learning_rate,
            decay_steps,
            decay_rate,
            staircase,
        }
    }

    pub fn get_learning_rate(&self, epoch: usize) -> f64 {
        let mut learning_rate =
            self.initial_learning_rate / (1. + self.decay_rate * (epoch as f64 / self.decay_steps));
        if self.staircase {
            learning_rate = self.initial_learning_rate
                / (1. + self.decay_rate * (epoch as f64 / self.decay_steps).floor());
        }
        learning_rate
    }
}
