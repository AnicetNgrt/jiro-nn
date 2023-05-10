use crate::vision::{
    conv_initializers::ConvInitializers, conv_optimizer::ConvOptimizers, image::Image,
    image::ImageTrait,
};

#[derive(Debug)]
pub struct DenseConvLayer {
    pub kernels: Image,
    pub biases: Image,
    pub input: Option<Image>,
    pub output: Option<Image>,
    kernels_optimizer: ConvOptimizers,
    biases_optimizer: ConvOptimizers,
}

impl DenseConvLayer {
    pub fn new(
        nrow: usize,
        ncol: usize,
        nchan: usize,
        nkern: usize,
        kernels_initializer: ConvInitializers,
        biases_initializer: ConvInitializers,
        kernels_optimizer: ConvOptimizers,
        biases_optimizer: ConvOptimizers,
    ) -> Self {
        Self {
            kernels: kernels_initializer.gen_image(nrow, ncol, nchan, nkern),
            biases: biases_initializer.gen_image(nrow, ncol, nchan, 1),
            input: None,
            output: None,
            kernels_optimizer,
            biases_optimizer,
        }
    }

    pub fn forward(&mut self, input: Image) -> Image {
        let res = input
            .cross_correlate(&self.kernels)
            .component_add(&self.biases);
        self.input = Some(input);
        self.output = Some(res.clone());
        res
    }

    pub fn backward(&mut self, epoch: usize, output_gradient: Image) -> Image {
        let input_grad = output_gradient.convolve_full(&self.kernels);

        let kern_grad = self
            .input
            .as_ref()
            .unwrap()
            .cross_correlate(&output_gradient);

        let mut biases_grad = output_gradient.get_channel(0);
        for c in 1..output_gradient.image_dims().2 {
            biases_grad = biases_grad.component_add(&output_gradient.get_channel(c));
        }

        self.kernels = self
            .kernels_optimizer
            .update_parameters(epoch, &self.kernels, &kern_grad);
        self.biases = self
            .biases_optimizer
            .update_parameters(epoch, &self.biases, &biases_grad);
        input_grad
    }
}
