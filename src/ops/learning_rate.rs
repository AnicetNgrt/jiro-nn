use crate::linalg::Scalar;

pub trait LearningRateScheduler<'opgraph> {
    fn increment_step(&mut self);
    fn get_learning_rate(&self) -> Scalar;
    fn clone_box(&self) -> Box<dyn LearningRateScheduler<'opgraph> + 'opgraph>;
}

pub struct ConstantLearningRate {
    learning_rate: Scalar,
}

impl ConstantLearningRate {
    pub fn new(learning_rate: Scalar) -> Self {
        Self { learning_rate }
    }
}

impl<'opgraph> LearningRateScheduler<'opgraph> for ConstantLearningRate {
    fn increment_step(&mut self) {}

    fn get_learning_rate(&self) -> Scalar {
        self.learning_rate
    }

    fn clone_box(&self) -> Box<dyn LearningRateScheduler<'opgraph> + 'opgraph> {
        Box::new(ConstantLearningRate::new(self.learning_rate))
    }
}
