use nn::data_utils::DataTable;
use polars::prelude::*;
use std::{error::Error};

fn main() -> Result<(), Box<dyn Error>> {
    // Load the CSV file into a DataFrame
    let file_path = "dataset/kc_house_data.csv";
    let df = CsvReader::from_path(file_path)?.finish()?;

    let mut normalized_df = DataTable(df)
        .normalize(Some(vec!["id", "date", "yr_built", "yr_renovated"]))
        .df();

    let mut file = std::fs::File::create("dataset/normalized.csv").unwrap();
    CsvWriter::new(&mut file).finish(&mut normalized_df).unwrap();

    Ok(())
}
