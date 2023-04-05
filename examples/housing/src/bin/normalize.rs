use nn::datatable::DataTable;

fn main() {
    DataTable::from_file("dataset/kc_house_data.csv")
        .auto_normalize(Some(&["id", "date", "yr_built", "yr_renovated"]))
        .to_file("dataset/normalized.csv");
}
