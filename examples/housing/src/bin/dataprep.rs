use housing::{FEATURES_NAMES};
use nn::datatable::DataTable;

const TRAINING_POINTS: usize = 17290;
const TESTING_POINTS: usize = 4322;

fn main() {
    let (mut train_data, mut test_data) = DataTable::from_file("dataset/normalized.csv")
        .sample(Some(TRAINING_POINTS + TESTING_POINTS), true)
        .select_columns(FEATURES_NAMES)
        .split(TRAINING_POINTS, TESTING_POINTS);

    train_data.to_file("dataset/train.csv");
    test_data.to_file("dataset/test.csv");
}
