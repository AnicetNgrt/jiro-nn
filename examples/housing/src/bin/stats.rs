use nn::datatable::DataTable;
use polars::prelude::*;

fn main() {
    let mut stats = DataTable(DataTable::from_file("dataset/normalized.csv").df().lazy()
        .select([
            col("price").min().alias("min_price"),
            col("price").max().alias("max_price"),
            col("bedrooms").min().alias("min_bdrs"),
            col("bedrooms").max().alias("max_bdrs"),
            col("lat").min().alias("min_lat"),
            col("lat").max().alias("max_lat"),
            col("long").min().alias("min_long"),
            col("long").max().alias("max_long"),
            col("sqft_lot").min().alias("min_sqft_lot"),
            col("sqft_lot").max().alias("max_sqft_lot")
        ])
        .collect().unwrap());

    stats.to_file("dataset/stats.csv");
}