use nn::datatable::DataTable;
use polars::prelude::*;

fn main() {
    let stats = DataTable::from_dataframe(
        DataTable::from_file("dataset/kc_house_data.csv")
            .map_f64_column("price", |x| x.log(10.))
            .df()
            .lazy()
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
                col("sqft_lot").max().alias("max_sqft_lot"),
            ])
            .collect()
            .unwrap(),
    );

    stats.to_file("dataset/stats.csv");
}
