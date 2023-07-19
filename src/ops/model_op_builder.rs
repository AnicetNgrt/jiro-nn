use super::{ModelOp, Data};

pub trait ModelOpBuilder<DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data> {
    fn build(self) -> dyn ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>;
}