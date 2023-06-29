use std::{
    collections::HashMap,
    sync::{
        mpsc::{self, Receiver, Sender},
        Mutex,
    },
    thread::{self, ThreadId},
    time::{Duration, Instant},
};

use lazy_static::lazy_static;

lazy_static! {
    static ref SENDER_TO_GI: Mutex<Sender<Messages>> = Mutex::new(GI::run());
}

#[derive(Clone)]
struct Task {
    name: String,
    start: Instant,
    end: Option<Instant>,
}

/// Global Introspector (GI)
/// Keeps track of tasks on another thread in order to monitor progress in real time.
pub struct GI {
    threads_stacks: HashMap<ThreadId, Vec<Task>>,
    thread_ids_order: Vec<ThreadId>,
    recently_finished_long_tasks: Vec<Task>,
}

#[derive(Clone)]
enum Messages {
    StartTask { name: String, thread_id: ThreadId },
    EndTask { thread_id: ThreadId },
    Stop,
}

impl GI {
    fn new() -> Self {
        Self {
            threads_stacks: HashMap::new(),
            thread_ids_order: Vec::new(),
            recently_finished_long_tasks: Vec::new(),
        }
    }

    fn pretty_print_tasks(&self) {
        // clear terminal
        print!("{esc}c", esc = 27 as char);

        // print recently finished tasks
        println!("Recently finished tasks (longer than 100ms):\n");
        let max_task_name_length = self
            .recently_finished_long_tasks
            .iter()
            .map(|task| task.name.len())
            .max()
            .unwrap_or(0);

        for task in &self.recently_finished_long_tasks {
            let duration = task.end.unwrap() - task.start;
            let formatted_task_name = format!("{:width$}", task.name, width = max_task_name_length);
            println!("- {}: ({:?})", formatted_task_name, duration);
        }

        // print current tasks
        println!("\nCurrent tasks:\n");
        for thread_id in &self.thread_ids_order {
            let stack = self.threads_stacks.get(thread_id).unwrap();
            let mut indent = 0;
            for task in stack {
                println!(
                    "└{} {}: ({:?})",
                    "─".repeat(indent),
                    task.name,
                    task.start.elapsed()
                );
                indent += 1;
            }
        }
    }

    fn _get_stack_or_create(&mut self, thread_id: ThreadId) -> &mut Vec<Task> {
        if !self.thread_ids_order.contains(&thread_id) {
            self.thread_ids_order.push(thread_id);
        }
        self.threads_stacks.entry(thread_id).or_insert(Vec::new())
    }

    fn _add_task(&mut self, thread_id: ThreadId, task: Task) {
        let stack = self._get_stack_or_create(thread_id);
        stack.push(task);
    }

    fn _update_current_task(&mut self, thread_id: ThreadId, task: Task) {
        let stack = self._get_stack_or_create(thread_id);
        stack.pop();
        stack.push(task);
    }

    fn _end_task(&mut self, thread_id: ThreadId) -> Option<Task> {
        let stack = self._get_stack_or_create(thread_id);
        let maybe_task = stack.pop();
        if let Some(mut task) = maybe_task.clone() {
            if task.start.elapsed() > Duration::from_millis(100) {
                task.end = Some(Instant::now());
                let full_name = stack
                    .iter()
                    .map(|t| t.name.clone())
                    .collect::<Vec<_>>()
                    .join(".");
                task.name = format!("{}.{}", full_name, task.name);
                self.recently_finished_long_tasks.push(task);

                if self.recently_finished_long_tasks.len() > 20 {
                    self.recently_finished_long_tasks.remove(0);
                }
            }
        }

        maybe_task
    }

    fn _get_current_task(&mut self, thread_id: ThreadId) -> Option<&Task> {
        let stack = self._get_stack_or_create(thread_id);
        stack.last()
    }

    fn introspect(mut self, rx: Receiver<Messages>) {
        loop {
            if let Ok(message) = rx.recv_timeout(Duration::from_millis(100)) {
                match message.clone() {
                    Messages::StartTask { name, thread_id } => {
                        self._add_task(
                            thread_id,
                            Task {
                                name,
                                start: Instant::now(),
                                end: None,
                            },
                        );
                    }
                    Messages::EndTask { thread_id } => {
                        let _ = self._end_task(thread_id).unwrap();
                    }
                    Messages::Stop => break,
                };
            }

            self.pretty_print_tasks()
        }
    }

    fn run() -> Sender<Messages> {
        let (tx, rx) = mpsc::channel::<Messages>();
        thread::spawn(move || {
            let gi = GI::new();
            gi.introspect(rx);
        });

        tx.clone()
    }

    fn send(message: Messages) {
        let sender = SENDER_TO_GI.lock().unwrap();
        sender.send(message).unwrap();
    }

    pub fn start_task<S: ToString>(name: S) {
        Self::send(Messages::StartTask {
            name: name.to_string(),
            thread_id: thread::current().id(),
        });
    }

    pub fn end_task() {
        Self::send(Messages::EndTask {
            thread_id: thread::current().id(),
        });
    }

    pub fn stop() {
        Self::send(Messages::Stop);
    }
}
