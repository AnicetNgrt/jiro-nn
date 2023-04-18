use std::collections::HashMap;

use plotters::{
    prelude::{
        BitMapBackend, ChartBuilder, Circle, IntoDrawingArea, IntoLinspace, Rectangle,
    },
    series::{LineSeries, SurfaceSeries},
    style::{Color, IntoFont, RGBColor, BLACK, WHITE},
};

use crate::vec_utils::{normalize_tensor, normalize_matrix};

pub enum YAxis {
    Discrete(Vec<f64>),
    Continuous(Box<dyn Fn(f64) -> f64>),
    Continuous2D(Box<dyn Fn(f64, f64) -> f64>),
}

pub struct Chart {
    x_axis: HashMap<String, Vec<f64>>,
    x_axis_names: Vec<String>,
    y_axis: HashMap<String, YAxis>,
    x_step: f64,
    y_step: f64,
    title: String,
    y_axis_label: String,
    colors: Vec<(u8, u8, u8)>,
}

impl Chart {
    pub fn new<S1: AsRef<str>, S2: AsRef<str>>(title: S1, y_axis_label: S2) -> Chart {
        Chart {
            x_axis: HashMap::new(),
            x_axis_names: Vec::new(),
            y_axis: HashMap::new(),
            x_step: 0.01,
            y_step: 0.01,
            title: title.as_ref().to_string(),
            y_axis_label: y_axis_label.as_ref().to_string(),
            colors: vec![
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
            ],
        }
    }

    pub fn set_x_step(&mut self, step: f64) -> &mut Self {
        self.x_step = step;
        self
    }

    pub fn set_y_step(&mut self, step: f64) -> &mut Self {
        self.y_step = step;
        self
    }

    pub fn add_range_x_axis(&mut self, label: &str, min: f64, max: f64, step: f64) -> &mut Self {
        self.set_x_step(step);
        self.x_axis.insert(
            label.to_string(),
            ((min / self.x_step) as usize..(max / self.x_step) as usize)
                .map(|x| (x as f64) * self.x_step)
                .collect::<Vec<f64>>(),
        );
        self.x_axis_names.push(label.to_string());
        self
    }

    pub fn add_x_axis(&mut self, label: &str, x: Vec<f64>) -> &mut Self {
        self.x_axis.insert(label.to_string(), x);
        self.x_axis_names.push(label.to_string());
        self
    }

    pub fn add_discrete_y(&mut self, label: &str, y: Vec<f64>) -> &mut Self {
        self.y_axis.insert(label.to_string(), YAxis::Discrete(y.clone()));
        self
    }

    pub fn add_continuous_y(&mut self, label: &str, y: &'static dyn Fn(f64) -> f64) -> &mut Self {
        self.y_axis.insert(
            label.to_string(),
            YAxis::Continuous(Box::new(y)),
        );
        self
    }

    pub fn add_continuous_2d_y(&mut self, label: &str, y: &'static dyn Fn(f64, f64) -> f64) -> &mut Self {
        self.y_axis.insert(
            label.to_string(),
            YAxis::Continuous2D(Box::new(y)),
        );
        self
    }

    fn first_x_name(&self) -> String {
        self.x_axis_names[0].to_string()
    }

    fn first_two_x_names(&self) -> (String, String) {
        (
            self.x_axis_names[0].to_string(),
            self.x_axis_names[1].to_string()
        )
    }

    fn y_names(&self) -> Vec<String> {
        self.y_axis.keys().map(|x| x.to_string()).collect()
    }

    fn is_y_discrete(&self, name: &str) -> bool {
        match self.y_axis.get(name).unwrap() {
            YAxis::Discrete(_) => true,
            _ => false,
        }
    }

    fn all_y_values(&self) -> Vec<Vec<f64>> {
        let mut values = Vec::new();

        for (_, axis) in self.y_axis.iter() {
            match axis {
                YAxis::Discrete(v) => {
                    values.push(v.clone());
                }
                YAxis::Continuous(generator) => {
                    let x_vals = self.x_axis.get(&self.first_x_name()).unwrap();
                    values.push(
                        x_vals
                            .iter()
                            .map(|x| generator(*x))
                            .collect::<Vec<f64>>(),
                    );
                },
                YAxis::Continuous2D(generator) => {
                    let (x1_name, x2_name) = self.first_two_x_names();
                    values.push(
                        self.x_axis.get(&x1_name).unwrap()
                            .iter()
                            .zip(self.x_axis.get(&x2_name).unwrap().iter())
                            .map(|(x1, x2)| generator(*x1, *x2))
                            .collect::<Vec<f64>>(),
                    );
                },
            }
        }

        values
    }

    pub fn scatter<S: AsRef<str>>(&self, file: S) {
        let x_name = self.first_x_name();
        let y_names = self.y_names();

        let root = BitMapBackend::new(file.as_ref(), (1224, 768)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let (x_vals, x_min, x_max) = normalize_tensor(self.x_axis.get(&x_name).unwrap());
        let (y_all_vals, y_min, y_max) = normalize_matrix(&self.all_y_values());

        let mut chart = ChartBuilder::on(&root)
            .caption(self.title.clone(), ("sans-serif", 20))
            .margin(10)
            .set_left_and_bottom_label_area_size(75)
            .build_cartesian_2d(
                (0f64..1f64).step(self.x_step),
                (0f64..1f64).step(self.y_step)
            )
            .unwrap();

        chart
            .configure_mesh()
            .x_label_formatter(&|v| format!("{:.4}", v * (x_max - x_min) + x_min))
            .x_desc(x_name)
            .y_desc(self.y_axis_label.clone())
            .y_label_formatter(&|v| format!("{:.4}", v * (y_max - y_min) + y_min))
            .disable_mesh()
            .draw()
            .unwrap();

        for (i, y_name) in y_names.iter().enumerate() {
            let color = self.colors[i % self.colors.len()];
            let color = RGBColor(color.0, color.1, color.2);

            let series_annot = if self.is_y_discrete(&y_name) {
                chart
                    .draw_series(
                        x_vals
                            .clone()
                            .into_iter()
                            .zip(y_all_vals[i].clone().into_iter())
                            .map(|(x, y)| {
                                //println!("{} {}", x, y);
                                Circle::new((x, y), 1.5, color.mix(0.8))
                            }),
                    )
                    .unwrap()
            } else {
                chart
                    .draw_series(
                        LineSeries::new(
                            x_vals
                                .clone()
                                .into_iter()
                                .zip(y_all_vals[i].clone().into_iter()),
                            &color,
                        )
                    )
                    .unwrap()
            };

            series_annot.label(&y_names[i])
                .legend(move |(x, y)| Circle::new((x, y), 3, &color));
        }

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .background_style(&WHITE.mix(0.8))
            .draw()
            .unwrap();

        root.present().unwrap();
    }

    pub fn scatter_3d<S: AsRef<str>>(&self, file: S) {
        let (x1_name, x2_name) = self.first_two_x_names();
        let y_names = self.y_names();

        let root = BitMapBackend::new(file.as_ref(), (1224, 768)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let (x1_vals, x_min, x_max) = normalize_tensor(self.x_axis.get(&x1_name).unwrap());
        let (x2_vals, z_min, z_max) = normalize_tensor(self.x_axis.get(&x2_name).unwrap());
        let (y_all_vals, y_min, y_max) = normalize_matrix(&self.all_y_values());

        let mut chart = ChartBuilder::on(&root)
            .caption(self.title.clone(), ("sans-serif", 20).into_font())
            .margin(10)
            .x_label_area_size(75)
            .y_label_area_size(75)
            .build_cartesian_3d(
                (0f64..1f64).step(self.x_step),
                (0f64..1f64).step(self.y_step),
                (0f64..1f64).step(self.x_step)
            )
            .unwrap();

        chart
            .configure_axes()
            .x_formatter(&|x| format!("{}={:.4}", x1_name, x * (x_max - x_min) + x_min))
            .y_formatter(&|y| format!("{}={:.4}", self.y_axis_label, y * (y_max - y_min) + y_min))
            .z_formatter(&|z| format!("{}={:.4}", x2_name, z * (z_max - z_min) + z_min))
            .draw()
            .unwrap();

        chart.with_projection(|mut pb| {
            pb.yaw = 0.5;
            pb.scale = 1.0;
            pb.pitch = 0.3;
            pb.into_matrix()
        });

        for (i, y_name) in y_names.iter().enumerate() {
            let color = self.colors[i % self.colors.len()];
            let color = RGBColor(color.0, color.1, color.2);

            let series_annot = if self.is_y_discrete(&y_name) {
                chart
                    .draw_series(
                        x1_vals
                            .clone()
                            .into_iter()
                            .zip(x2_vals.clone().into_iter())
                            .zip(y_all_vals[i].clone().into_iter())
                            .map(|((x1, x2), y)| {
                                //println!("{} {}", x, y);
                                Circle::new((x1, y, x2), 1.5, color.mix(0.8))
                            }),
                    )
                    .unwrap()
            } else {
                chart
                    .draw_series(
                        SurfaceSeries::xoz(
                            x1_vals.clone().into_iter(),
                            x2_vals.clone().into_iter(),
                            |x, _| {
                                let ix = (x * (x1_vals.len() - 1) as f64).round() as usize;
                                y_all_vals[i][ix]
                            },
                        )
                        .style(color.filled())
                    )
                    .unwrap()
            };
            series_annot.label(y_names[i].clone())
                .legend(move |(x, y)| {
                    Rectangle::new([(x + 5, y - 5), (x + 15, y + 5)], color.filled())
                });
        }

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .background_style(&WHITE.mix(0.8))
            .draw()
            .unwrap();

        root.present().unwrap();
    }
}
