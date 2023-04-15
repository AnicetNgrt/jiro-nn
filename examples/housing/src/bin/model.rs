use housing::{FEATURES, OUT};
use nn::benchmarking::{EpochEvaluation, FoldEvaluation, ModelEvaluation};
use nn::datatable::DataTable;
use nn::model_spec::ModelSpec;
use nn::optimizer::Optimizers;
use nn::pipelines::attach_ids::AttachIds;
use nn::pipelines::extract_months::ExtractMonths;
use nn::pipelines::extract_timestamps::ExtractTimestamps;
use nn::pipelines::log_scale::LogScale10;
use nn::pipelines::normalize::Normalize;
use nn::pipelines::Pipeline;
use nn::{activation::Activation, network::Network, nn};

pub fn new_network(
    hidden_sizes: Vec<usize>,
    activation: Activation,
    optimizer: Optimizers,
) -> Network {
    let mut sizes = vec![FEATURES];
    for s in hidden_sizes {
        sizes.push(s);
    }
    sizes.push(OUT);
    nn(sizes, vec![activation], vec![optimizer])
}

pub fn smoothstep(x: f64, min: f64, max: f64) -> f64 {
    if x <= min {
        0.
    } else if x >= max {
        1.
    } else {
        (x - min) / (max - min)
    }
}

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];

    let model = ModelSpec::from_json_file(format!("models/{}.json", config_name).as_str());
    println!("model: {:#?}", model);

    let mut pipeline = Pipeline::new();
    let (updated_dataset_spec, data) = pipeline
        .add(AttachIds::new("id"))
        .add(ExtractMonths)
        .add(ExtractTimestamps)
        .add(LogScale10)
        .add(Normalize::new())
        .run("dataset", &model.dataset);

    println!("dataset: {:#?}", updated_dataset_spec);
    println!("data: {:#?}", data);

    let model = model.with_new_dataset(updated_dataset_spec);

    let mut validation_preds = DataTable::new_empty();

    let mut model_eval = ModelEvaluation::new_empty();
    for i in 0..model.folds {
        let mut network = model.to_network();
        println!("j: {} i: {}", network.j, network.i);

        let (train, validation) = data.split_k_folds(model.folds, i);

        let (validation_x_table, validation_y_table) =
            validation.random_order_in_out(&model.dataset.out_features_names());

        let validation_x = validation_x_table.drop_column("id").to_tensors();
        let validation_y = validation_y_table.to_tensors();

        let mut fold_eval = FoldEvaluation::new_empty();
        for e in 0..model.epochs {
            let (train_x_table, train_y_table) =
                train.random_order_in_out(&model.dataset.out_features_names());

            let train_x = train_x_table.drop_column("id").to_tensors();
            let train_y = train_y_table.to_tensors();

            let train_loss = network.train(
                e,
                &train_x,
                &train_y,
                &model.loss.to_loss(),
                model.batch_size.unwrap_or(train_x.len()),
            );

            let (preds, loss_avg, loss_std) =
                network.predict_evaluate_many(&validation_x, &validation_y, &model.loss.to_loss());

            println!(
                "Fold {:3} Epoch {:4} Train avg loss: {:.6} Pred avg loss: {:.6}",
                i, e, train_loss, loss_avg
            );

            let eval = EpochEvaluation::new(train_loss, loss_avg, loss_std);

            if e == model.epochs - 1 {
                validation_preds = validation_preds.apppend(
                    &DataTable::from_tensors(&model.dataset.out_features_names(), &preds)
                        .add_column_from(&validation_x_table, "id"),
                )
            };

            fold_eval.add_epoch(eval);
        }
        model_eval.add_fold(fold_eval);
    }

    // let out_features_names = &model.dataset.out_features_names();
    // let pred_out_features_names = &model.dataset.pred_out_features_names();

    let validation_preds = pipeline.revert_columnswise(&validation_preds);
    let data = pipeline.revert_columnswise(&data);
    let data_and_preds = data.inner_join(&validation_preds, "id", "id", Some("pred"));

    data_and_preds.to_file(format!("models_stats/{}.csv", config_name).as_str());

    println!("{:#?}", data_and_preds);

    model_eval.to_json_file(format!("models_stats/{}.json", config_name).as_str());
}
