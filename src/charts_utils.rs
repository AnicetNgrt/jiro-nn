use std::collections::HashMap;

use plotters::{
    prelude::{
        BitMapBackend, ChartBuilder, Circle, IntoDrawingArea, IntoLinspace, PathElement, Rectangle,
    },
    series::{LineSeries, SurfaceSeries},
    style::{Color, IntoFont, BLACK, WHITE, RGBColor},
};

use crate::vec_utils::{max_vecf64, max_vecvecf64, min_vecf64, min_vecvecf64};

pub struct Chart {
    x_axis: HashMap<String, Vec<f64>>,
    y_axis: HashMap<String, Vec<f64>>,
    x_step: f64,
    y_step: f64,
    title: String,
    y_axis_label: String,
    colors: Vec<(u8, u8, u8)>,
}

impl Chart {
    pub fn new(title: &str, y_axis_label: &str) -> Chart {
        Chart {
            x_axis: HashMap::new(),
            y_axis: HashMap::new(),
            x_step: 0.01,
            y_step: 0.01,
            title: title.to_string(),
            y_axis_label: y_axis_label.to_string(),
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
        self
    }

    pub fn add_x_axis(&mut self, label: &str, x: Vec<f64>) -> &mut Self {
        self.x_axis.insert(label.to_string(), x);
        self
    }

    pub fn add_y_axis(&mut self, label: &str, y: Vec<f64>) -> &mut Self {
        self.y_axis.insert(label.to_string(), y);
        self
    }

    fn first_x_name(&self) -> String {
        self.x_axis.keys().next().unwrap().to_string()
    }

    fn first_two_x_names(&self) -> (String, String) {
        let mut iter = self.x_axis.keys();
        let first = iter.next().unwrap().to_string();
        let second = iter.next().unwrap().to_string();
        (first, second)
    }

    fn y_names(&self) -> Vec<String> {
        self.y_axis.keys().map(|x| x.to_string()).collect()
    }

    pub fn scatter_plot(&self, file: &str) {
        let x_name = self.first_x_name();
        let y_names = self.y_names();

        let root = BitMapBackend::new(file, (1024, 768)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let x_vals = self.x_axis.get(&x_name).unwrap();
        let y_all_vals: Vec<_> = y_names
            .iter()
            .map(|y_name| self.y_axis.get(y_name).unwrap().clone())
            .collect();

        let x_min = min_vecf64(&x_vals);
        let x_max = max_vecf64(&x_vals);
        println!("x_min: {}, x_max: {}", x_min, x_max);
        println!("x_vals: {:?}", x_vals);

        let y_min = min_vecvecf64(&y_all_vals);
        let y_max = max_vecvecf64(&y_all_vals);

        let x_axis = (x_min..x_max).step(self.x_step);
        let y_axis = (y_min..y_max).step(self.y_step);

        let mut chart = ChartBuilder::on(&root)
            .caption(self.title.clone(), ("sans-serif", 20))
            .margin(10)
            .set_left_and_bottom_label_area_size(75)
            .build_cartesian_2d(x_axis.clone(), y_axis.clone())
            .unwrap();

        chart
            .configure_mesh()
            .x_label_formatter(&|v| format!("{:.0}", v))
            .x_desc(x_name)
            .y_desc(self.y_axis_label.clone())
            .disable_mesh()
            .draw()
            .unwrap();

        for i in 0..y_all_vals.len() {
            let color = self.colors[i % self.colors.len()];
            let color = RGBColor(color.0, color.1, color.2);

            chart
                .draw_series(
                    x_vals
                        .clone()
                        .into_iter()
                        .zip(y_all_vals[i].clone().into_iter())
                        .map(|(x, y)| {
                            //println!("{} {}", x, y);
                            Circle::new((x, y), 3, &color)
                        }),
                )
                .unwrap()
                .label(&y_names[i])
                .legend(move |(x, y)| Circle::new((x, y), 3, &color));
        }

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()
            .unwrap();

        root.present().unwrap();
    }

    pub fn line_plot(&self, file: &str) {
        let x_name = self.first_x_name();
        let y_names = self.y_names();

        let root = BitMapBackend::new(file, (1024, 768)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let x_vals = self.x_axis.get(&x_name).unwrap();
        let y_all_vals: Vec<_> = y_names
            .iter()
            .map(|y_name| self.y_axis.get(y_name).unwrap().clone())
            .collect();

        let x_min = min_vecf64(&x_vals);
        let x_max = max_vecf64(&x_vals);

        let y_min = min_vecvecf64(&y_all_vals);
        let y_max = max_vecvecf64(&y_all_vals);

        let x_axis = (x_min..x_max).step(self.x_step);
        let y_axis = (y_min..y_max).step(self.y_step);

        let mut chart = ChartBuilder::on(&root)
            .caption(self.title.clone(), ("sans-serif", 20))
            .margin(10)
            .set_left_and_bottom_label_area_size(75)
            .build_cartesian_2d(x_axis.clone(), y_axis.clone())
            .unwrap();

        chart
            .configure_mesh()
            .x_label_formatter(&|v| format!("{:.0}", v))
            .x_desc(x_name)
            .y_desc(self.y_axis_label.clone())
            .disable_mesh()
            .draw()
            .unwrap();

        for i in 0..y_all_vals.len() {
            let color = self.colors[i % self.colors.len()];
            let color = RGBColor(color.0, color.1, color.2);

            chart
                .draw_series(LineSeries::new(
                    x_vals
                        .clone()
                        .into_iter()
                        .zip(y_all_vals[i].clone().into_iter()),
                    &color,
                ))
                .unwrap()
                .label(&y_names[i])
                .legend(move |(x, y)| PathElement::new([(x, y), (x + 15, y)], &color));
        }

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()
            .unwrap();

        root.present().unwrap();
    }

    pub fn surface_plot(&self, file: &str) {
        let (x1_name, x2_name) = self.first_two_x_names();
        let y_names = self.y_names();

        let root = BitMapBackend::new(file, (1024, 768)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let x1_vals = self.x_axis.get(&x1_name).unwrap();
        let x2_vals = self.x_axis.get(&x2_name).unwrap();
        let y_all_vals: Vec<_> = y_names
            .iter()
            .map(|y_name| self.y_axis.get(y_name).unwrap().clone())
            .collect();

        let x_min = min_vecf64(&x1_vals);
        let z_min = min_vecf64(&x2_vals);
        let x_max = min_vecf64(&x1_vals);
        let z_max = min_vecf64(&x2_vals);

        let y_min = min_vecvecf64(&y_all_vals);
        let y_max = max_vecvecf64(&y_all_vals);

        let x_axis = (x_min..x_max).step(self.x_step);
        let y_axis = (y_min..y_max).step(self.y_step);
        let z_axis = (z_min..z_max).step(self.x_step);

        let x_z_to_y_hashmap = x1_vals
            .iter()
            .zip(x2_vals.iter())
            .zip(y_all_vals.iter())
            .map(|((x, z), y)| (format!("{}-{}", x, z), y.clone()))
            .collect::<HashMap<String, Vec<f64>>>();

        let mut chart = ChartBuilder::on(&root)
            .caption(self.title.clone(), ("sans-serif", 50).into_font())
            .margin(10)
            .x_label_area_size(75)
            .y_label_area_size(75)
            .build_cartesian_3d(x_axis.clone(), y_axis.clone(), z_axis.clone())
            .unwrap();

        chart
            .configure_axes()
            .x_formatter(&|x| format!("{}={:.2}", x1_name, x))
            .y_formatter(&|y| format!("{}={:.2}", self.y_axis_label, y))
            .z_formatter(&|z| format!("{}={:.2}", x2_name, z))
            .draw()
            .unwrap();

        for i in 0..y_names.len() {
            let color = self.colors[i % self.colors.len()];
            let color = RGBColor(color.0, color.1, color.2);

            chart
                .draw_series(
                    SurfaceSeries::xoz(
                        x1_vals.clone().into_iter(),
                        x2_vals.clone().into_iter(),
                        |x, z| x_z_to_y_hashmap.get(&format!("{}-{}", x, z)).unwrap()[i],
                    )
                    .style(color.filled()),
                )
                .unwrap()
                .label(y_names[i].clone())
                .legend(move |(x, y)| {
                    Rectangle::new([(x + 5, y - 5), (x + 15, y + 5)], color.filled())
                });
        }

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()
            .unwrap();

        root.present().unwrap();
    }
}
