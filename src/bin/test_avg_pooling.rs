use arrayfire::print;
use neural_networks_rust::vision::{
    conv_layer::avg_pooling_layer::AvgPoolingLayer, image::Image, image::ImageTrait,
    image_layer::ImageLayer,
};

pub fn main() {
    let image = Image::random_normal(6, 6, 3, 2, 2.0, 1.0);

    print(&image.0);

    let mut layer = AvgPoolingLayer::new(2);

    let image = layer.forward(image);

    print(&image.0);

    let gradient = Image::random_normal(3, 3, 3, 2, 0.0, 0.5);

    print(&gradient.0);

    let gradient = layer.backward(0, gradient);

    print(&gradient.0);
}
