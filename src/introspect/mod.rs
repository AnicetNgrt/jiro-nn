use std::{
    collections::{HashMap, HashSet},
    sync::{
        mpsc::{self, Receiver, Sender},
        Mutex,
    },
    thread,
    time::Instant,
};

use lazy_static::lazy_static;

lazy_static! {
    static ref SENDER_TO_GI: Mutex<Sender<Messages>> = Mutex::new(GlobalIntrospector::run());
}

pub struct GlobalIntrospector;

#[derive(Eq, Hash, PartialEq)]
enum ReportMode {
    Indicatif,
    Print,
}

enum Messages {
    ToggleReporting(ReportMode),
    StartProgress { name: String, max: usize },
    IncProgress { name: String, val: usize },
    AbortProgress { name: String },
    StartTask { name: String },
    EndTask { name: String },
    Stop,
}

fn introspector(rx: Receiver<Messages>) {
    let mut tasks: HashMap<String, Instant> = HashMap::new();
    let mut progresses: HashMap<String, (usize, usize)> = HashMap::new();
    let mut report_modes = HashSet::<ReportMode>::new();

    loop {
        let mut message = None;
        let can_print = report_modes.contains(&ReportMode::Print);

        match rx.recv().unwrap() {
            Messages::Stop => break,
            Messages::ToggleReporting(val) => {
                if report_modes.contains(&val) {
                    report_modes.remove(&val);
                } else {
                    report_modes.insert(val);
                }
            }
            Messages::StartProgress { name, max } => {
                progresses.insert(name.clone(), (0, max));
                message = Some(format!("[{}] Progress... {}/{}", name, 0, max));
            }
            Messages::IncProgress { name, val } => {
                let progress = progresses.get_mut(&name).unwrap();
                let (old_progress, max) = progress;
                *progress = (*old_progress + val, *max);

                if progress.0 < progress.1 {
                    message = Some(format!(
                        "[{}] Progress... {}/{}",
                        name, progress.0, progress.1
                    ));
                } else {
                    message = Some(format!(
                        "[{}] Completed {}/{}",
                        name, progress.0, progress.1
                    ));
                }
            }
            Messages::AbortProgress { name } => {
                progresses.remove(&name);
                message = Some(format!("[{}] Aborted", name));
            }
            Messages::StartTask { name } => {
                tasks.insert(name.clone(), Instant::now());
                message = Some(format!("[{}] Started", name));
            }
            Messages::EndTask { name } => {
                let start = tasks.get(&name).unwrap();

                message = Some(format!(
                    "[{}] Ended after {}s or {}ms",
                    name,
                    Instant::elapsed(&start).as_secs(),
                    Instant::elapsed(&start).as_millis()
                ));
            }
        }

        if let Some(message) = message {
            if can_print {
                println!("{}", message);
            }
        }
    }
}

impl GlobalIntrospector {
    fn run() -> Sender<Messages> {
        let (tx, rx) = mpsc::channel::<Messages>();
        thread::spawn(move || introspector(rx));

        tx.clone()
    }

    fn send(message: Messages) {
        let sender = SENDER_TO_GI.lock().unwrap();
        sender.send(message).unwrap();
    }

    pub fn toggle_printing() {
        Self::send(Messages::ToggleReporting(ReportMode::Print));
    }

    pub fn toggle_progress_bars() {
        Self::send(Messages::ToggleReporting(ReportMode::Indicatif));
    }

    pub fn start_progress<S: ToString>(name: S, max: usize) {
        Self::send(Messages::StartProgress {
            name: name.to_string(),
            max,
        });
    }

    pub fn inc_progress<S: ToString>(name: S, val: usize) {
        Self::send(Messages::IncProgress {
            name: name.to_string(),
            val,
        });
    }

    pub fn abort_progress<S: ToString>(name: S) {
        Self::send(Messages::AbortProgress {
            name: name.to_string(),
        });
    }

    pub fn start_task<S: ToString>(name: S) {
        Self::send(Messages::StartTask {
            name: name.to_string(),
        });
    }

    pub fn end_task<S: ToString>(name: S) {
        Self::send(Messages::EndTask {
            name: name.to_string(),
        });
    }

    pub fn stop() {
        Self::send(Messages::Stop);
    }
}
