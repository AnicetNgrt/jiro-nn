use serde::{Serialize, Deserialize};


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiecewiseConstant {
    pub boundaries: Vec<usize>,
    pub values: Vec<f64>,
}

impl PiecewiseConstant {
    pub fn new(boundaries: Vec<usize>, values: Vec<f64>) -> Self {
        Self { boundaries, values }
    }

    pub fn get_learning_rate(&self, epoch: usize) -> f64 {
        let mut learning_rate = self.values[0];
        for (i, boundary) in self.boundaries.iter().enumerate() {
            if epoch >= *boundary {
                learning_rate = self.values[i + 1];
            }
        }
        learning_rate
    }
}