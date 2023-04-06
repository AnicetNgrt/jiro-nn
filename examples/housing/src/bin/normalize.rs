use nn::datatable::DataTable;

fn main() {
    let mut dt = DataTable::from_file("dataset/kc_house_data.csv")
        .map_f64_column("price", |x| x.log(10.));
    
    println!("{:?}", dt.clone().df().column("price"));

    dt.auto_normalize(Some(&["id", "date"]))
        .to_file("dataset/normalized.csv");
}
