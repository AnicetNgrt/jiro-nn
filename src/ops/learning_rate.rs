use crate::linalg::Scalar;

pub trait LearningRateScheduler {
    fn increment_step(&mut self);
    fn get_learning_rate(&self) -> Scalar;
    fn clone_box(&self) -> Box<dyn LearningRateScheduler>;
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

    fn clone_box(&self) -> Box<dyn LearningRateScheduler> {
        Box::new(ConstantLearningRate::new(self.learning_rate))
    }
}
