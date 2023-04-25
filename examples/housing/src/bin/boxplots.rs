use gnuplot::{
    AutoOption::{Fix},
    AxesCommon, Coordinate, Figure,
    LabelOption::Rotate,
    PlotOption::{Color, PointSymbol}, MarginSide::MarginBottom,
};
use neural_networks_rust::{
    model::Model,
    pipelines::{
        extract_months::ExtractMonths, extract_timestamps::ExtractTimestamps, normalize::Normalize, Pipeline,
    },
    vec_utils::{vector_boxplot},
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];

    let model = Model::from_json_file(format!("models/{}.json", config_name));
    println!("model: {:#?}", model);

    let mut pipeline = Pipeline::new();
    let (_, data_before) = pipeline
        .add(ExtractMonths)
        .add(ExtractTimestamps)
        .add(Normalize::new())
        .run("./dataset", &model.dataset);

    let mut pipeline = Pipeline::basic_single_pass();
    let (after_spec, data) = pipeline
        .run("./dataset", &model.dataset);

    println!("{:#?}", data);

    let mut fg = Figure::new();

    let mut axes = fg.axes2d().set_title("Before and after preprocessing features boxes & whiskers", &[]).set_margins(&[
        MarginBottom(0.2)
    ]);

    for (i, feature_name) in after_spec.feature_names().iter().enumerate() {
        for (j, (prefix, data)) in vec![("before", &data_before), ("after", &data)]
            .iter()
            .enumerate()
        {
            if data.has_column(feature_name) == false {
                continue;
            }
            let vals = data.column_to_vector(&feature_name);
            let (q1, q2, q3, min, max) = vector_boxplot(&vals);
            let outliers = vals.into_iter()
                .filter(|x| *x < min || *x > max)
                .collect::<Vec<_>>();

            let color = if j == 0 { "red" } else { "blue" };
            axes = axes
                .label(
                    &format!("{} {}", prefix, feature_name.replace("_", " ")),
                    Coordinate::Axis(((i * 2) + j) as f64 + 0.1),
                    Coordinate::Axis(-0.02),
                    &[Rotate(-45.0)],
                )
                .box_and_whisker_set_width(
                    [((i * 2) + j) as f32 + 0.1].iter(),
                    [q1].iter(),
                    [min].iter(),
                    [q2 + 0.003].iter(),
                    [q2 + 0.003].iter(),
                    [0.4f32].iter(),
                    &[Color(color)],
                )
                .box_and_whisker_set_width(
                    [((i * 2) + j) as f32 + 0.1].iter(),
                    [q2 - 0.003].iter(),
                    [q2 - 0.003].iter(),
                    [max].iter(),
                    [q3].iter(),
                    [0.4f32].iter(),
                    &[Color(color)],
                );
            
            if outliers.len() > 0 {
                axes = axes.points(
                    vec![((i * 2) + j) as f64 + 0.1; outliers.len()], 
                    outliers, 
                    &[Color(color), PointSymbol('*')]
                );
            }
        }
    }

    axes.set_x_ticks(None, &[], &[])
        .set_y_ticks(Some((Fix(0.1), 10)), &[], &[])
        .set_y_grid(true)
        .set_x_range(Fix(-0.2), Fix((after_spec.feature_names().len() * 2) as f64))
        .set_y_range(Fix(0.0), Fix(1.0));

    fg.save_to_png(format!("visuals/{}_boxplots.png", config_name), 2448, 1224)
        .unwrap();
}
