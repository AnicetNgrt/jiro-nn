use jiro_nn::{loss::Losses, model::network_model::NetworkModelBuilder};

fn main() {
    let training_data_in = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];

    let training_data_out = vec![
        vec![0.0], 
        vec![1.0], 
        vec![1.0], 
        vec![0.0]
    ];

    let in_size = 2;
    let hidden_out_size = 3;
    let out_size = 1;

    let network_model = NetworkModelBuilder::new()
        .full_dense(hidden_out_size)
            .init_glorot_uniform()
            .sgd()
            .tanh()
        .end()
        .full_dense(out_size)
            .init_glorot_uniform()
            .sgd()
            .tanh()
        .end()
    .build();

    let mut network = network_model.to_network(in_size);

    let loss = Losses::MSE.to_loss();
    let batch_size = 1;

    for epoch in 0..50000 {
        let error = network.train(
            epoch,
            &training_data_in,
            &training_data_out,
            &loss,
            batch_size,
        );

        if epoch % 1000 == 0 {
            println!("Epoch: {} Average training loss: {}", epoch, error);
        }
    }
}
