use serde::{Serialize, Deserialize};

use crate::linalg::Scalar;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiecewiseConstant {
    pub boundaries: Vec<usize>,
    pub values: Vec<Scalar>,
}

impl PiecewiseConstant {
    pub fn new(boundaries: Vec<usize>, values: Vec<Scalar>) -> Self {
        Self { boundaries, values }
    }

    pub fn get_learning_rate(&self, epoch: usize) -> Scalar {
        let mut learning_rate = self.values[0];
        for (i, boundary) in self.boundaries.iter().enumerate() {
            if epoch >= *boundary {
                learning_rate = self.values[i + 1];
            }
        }
        learning_rate
    }
}