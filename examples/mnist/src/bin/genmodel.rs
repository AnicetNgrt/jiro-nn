use neural_networks_rust::{
    activation::Activation::*,
    dataset::{Dataset, FeatureOptions::*},
    model::{
        conv_network_spec::ConvNetworkLayerSpecTypes::*,
        conv_network_spec::ConvNetworkSpec,
        dense_conv_layer_spec::DenseConvLayerOptions::*,
        dense_conv_layer_spec::DenseConvLayerSpec,
        direct_conv_layer_spec::{DirectConvLayerSpec, DirectConvLayerOptions::*},
        full_conv_layer_spec::{ConvLayerSpecTypes::*, FullConvLayerSpec, FullConvLayerOptions::*},
        full_layer_spec::{FullLayerOptions::*, FullLayerSpec},
        LayerSpecTypes::*,
        Model,
        ModelOptions::*,
    },
    vision::{image_activation::ConvActivation::*, conv_optimizer::*},
    optimizer::momentum,
    trainers::Trainers,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];

    let mut dataset_spec = Dataset::from_csv("dataset/train.csv");
    dataset_spec
        .add_opt(Normalized.except(&["label"]))
        .add_opt_to("label", Out)
        .add_opt_to("label", OneHotEncode);

    let dropout = None;

    let conv_layers = vec![
        FullConv(FullConvLayerSpec::from_options(&[
            ConvActivation(ConvSigmoid),
            ConvDropout(dropout),
            Conv(Dense(DenseConvLayerSpec::from_options(&[
                KernelCount(32),
                KernelSize(3, 3),
                ConvOptimizer(conv_momentum()),
            ]))),
        ])),
        AvgPooling(2),
        FullConv(FullConvLayerSpec::from_options(&[
            ConvActivation(ConvSigmoid),
            ConvDropout(dropout),
            Conv(Direct(DirectConvLayerSpec::from_options(&[
                DCvKernelSize(5, 5),
                DCvOptimizer(conv_momentum()),
            ]))),
        ])),
        AvgPooling(3),
    ];

    let mut layers = vec![];
    layers.push(ConvNetwork(ConvNetworkSpec {
        layers: conv_layers,
        in_channels: 1,
        height: 28,
        width: 28,
    }));

    layers.push(Full(FullLayerSpec::from_options(&[
        OutSize(128),
        Activation(ReLU),
        Dropout(dropout),
        Optimizer(momentum()),
    ])));

    let final_layer = Full(FullLayerSpec::from_options(&[
        OutSize(10),
        Activation(Linear),
        Dropout(dropout),
        Optimizer(momentum()),
    ]));

    let model = Model::from_options(&[
        Dataset(dataset_spec),
        HiddenLayers(layers.as_slice()),
        FinalLayer(final_layer),
        BatchSize(240),
        Trainer(Trainers::SplitTraining(0.8)),
        Epochs(10),
    ]);

    //println!("{:#?}", model);

    model.to_json_file(format!("models/{}.json", config_name));
}
