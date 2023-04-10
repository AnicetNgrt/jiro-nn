// use chrono::{NaiveDateTime, Utc, DateTime, Datelike};
// use nn::datatable::DataTable;

fn main() {
    // let mut dt = DataTable::from_file("dataset/kc_house_data.csv")
    //     .map_f64_column("price", |x| x.log(10.))
    //     .map_str_column_to_f64_column("date", "timestamp", |date| {
    //         let datetime = NaiveDateTime::parse_from_str(date, "%Y%m%dT%H%M%S").unwrap();
    //         let timestamp: DateTime<Utc> = DateTime::from_utc(datetime, Utc);
    //         let unix_seconds = timestamp.timestamp();
    //         unix_seconds as f64
    //     })
    //     .map_str_column_to_f64_column("date", "month", |date| {
    //         let datetime = NaiveDateTime::parse_from_str(date, "%Y%m%dT%H%M%S").unwrap();
    //         datetime.month() as f64
    //     });
    
    // println!("{:?}", dt.clone().df().column("sqft_lot"));

    // dt.auto_normalize(Some(&["id", "date"]))
    //     .to_file("dataset/normalized.csv");
}
