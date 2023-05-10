pub mod image;
pub mod conv_layer;
pub mod conv_activation;
pub mod conv_initializers;
pub mod conv_optimizer;
pub mod conv_network;

#[cfg(feature = "arrayfire")]
pub mod arrayfire_image;
#[cfg(feature = "arrayfire")]
pub type Image = arrayfire_image::Image;