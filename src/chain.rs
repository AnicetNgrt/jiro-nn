use nalgebra::SVector;

use crate::layer::Layer;

pub struct Chain<
    const IN: usize,
    const H_IN: usize,
    const H_OUT: usize,
    const OUT: usize,
    ToHidden: Layer<IN, H_IN>,
    Hidden: Layer<H_IN, H_OUT>,
    ToOutput: Layer<H_OUT, OUT>,
> {
    to_hidden: ToHidden,
    hidden: Hidden,
    to_output: ToOutput,
}

impl<
        const IN: usize,
        const H_IN: usize,
        const H_OUT: usize,
        const OUT: usize,
        ToHidden: Layer<IN, H_IN>,
        Hidden: Layer<H_IN, H_OUT>,
        ToOutput: Layer<H_OUT, OUT>,
    > Layer<IN, OUT> for Chain<IN, H_IN, H_OUT, OUT, ToHidden, Hidden, ToOutput>
{
    fn forward(&mut self, input: nalgebra::SVector<f64, IN>) -> SVector<f64, OUT> {
        let hidden_in = self.to_hidden.forward(input);
        let hidden_output = self.hidden.forward(hidden_in);
        self.to_output.forward(hidden_output)
    }

    fn backward(
        &mut self,
        output_gradient: nalgebra::SVector<f64, OUT>,
        learning_rate: f64,
    ) -> SVector<f64, IN> {
        let hidden_gradient_out = self.to_output.backward(output_gradient, learning_rate);
        let hidden_gradient_in = self.hidden.backward(hidden_gradient_out, learning_rate);
        self.to_hidden.backward(hidden_gradient_in, learning_rate)
    }
}
