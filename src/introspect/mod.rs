use std::{time::Instant, sync::{mpsc::{self, Sender}, Mutex}, thread};

use lazy_static::lazy_static;
use polars::export::ahash::HashMap;


lazy_static! {
    static ref SENDER_TO_GI: Mutex<Sender<Messages>> = Mutex::new(GlobalIntrospector::run());
}

pub struct GlobalIntrospector {
    pub tasks: HashMap<String, Instant>,
    pub progresses: HashMap<String, (usize, usize)>
}

enum Messages {
    StartProgress { name: String, max: usize },
    IncProgress { name: String, val: usize },
    AbortProgress { name: String },
    StartTask { name: String },
    EndTask { name: String },
    Stop
}

impl GlobalIntrospector {
    fn run() -> Sender<Messages> {
        let (tx, rx) = mpsc::channel::<Messages>();
        thread::spawn(move || {
            loop {
                match rx.recv().unwrap() {
                    Messages::Stop => break,
                    Messages::StartProgress { name, max } => todo!(),
                    Messages::IncProgress { name, val } => todo!(),
                    Messages::AbortProgress { name } => todo!(),
                    Messages::StartTask { name } => todo!(),
                    Messages::EndTask { name } => todo!(),
                }
            }
        });
        
        tx.clone()
    }

    fn send(message: Messages) {
        let sender = SENDER_TO_GI.lock().unwrap();
        sender.send(message).unwrap();
    }

    pub fn start_progress<S: ToString>(name: S, max: usize) {
        Self::send(Messages::StartProgress { name: name.to_string(), max });
    }

    pub fn inc_progress<S: ToString>(name: S, val: usize) {
        Self::send(Messages::IncProgress { name: name.to_string(), val });
    }

    pub fn abort_progress<S: ToString>(name: S) {
        Self::send(Messages::AbortProgress { name: name.to_string() });
    }

    pub fn start_task<S: ToString>(name: S) {
        Self::send(Messages::StartTask { name: name.to_string() });
    }

    pub fn end_task<S: ToString>(name: S) {
        Self::send(Messages::EndTask { name: name.to_string() });
    }

    pub fn stop() {
        Self::send(Messages::Stop);
    }
}