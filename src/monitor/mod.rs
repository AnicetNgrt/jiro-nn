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
    static ref SENDER_TO_GI: Mutex<Sender<Messages>> = Mutex::new(TM::run());
}

const TERMINAL_HEIGHT: usize = 40;

#[derive(Clone)]
struct Task {
    name: String,
    start: Instant,
    end: Option<Instant>,
}

/// The Tasks Monitor (TM) keeps track of tasks in another thread in order to measure and report their progress in real time.
/// 
/// It displays the progress in a TUI. Therefore once activated prints to the terminal are not visible anymore. Other options will come later. 
/// 
/// It stacks the tasks so that every task has its own parent task.
/// 
/// The tasks stacks from different threads are separated. One thread's task cannot be a parent of another thread's task.
/// 
/// ```rust
/// TM::start_monitoring(); 
/// TM::start("task1");
/// // do something long
/// TM::end();
/// TM::start("task2");
/// // do something long
/// TM::start("subtask"); // creating the child task of task2
/// // do something long
/// TM::end(); // ending subtask
/// TM::start("subtask2"); // you can create subtasks, but only one at a given time
/// // do something long
/// TM::end(); // ending subtask2
/// TM::end_with_message("Task 2 and its subtasks are finished :-)"); // ending task2
/// TM::stop_monitoring();
/// ```
pub struct TM {
    threads_stacks: HashMap<ThreadId, Vec<Task>>,
    thread_ids_order: Vec<ThreadId>,
    recently_finished_long_tasks: Vec<Task>,
    recently_finished_task_message: Vec<String>,
}

#[derive(Clone)]
enum Messages {
    Start,
    StartTask {
        name: String,
        thread_id: ThreadId,
    },
    EndTask {
        thread_id: ThreadId,
        message: Option<String>,
    },
    Stop,
}

impl TM {
    fn new() -> Self {
        Self {
            threads_stacks: HashMap::new(),
            thread_ids_order: Vec::new(),
            recently_finished_long_tasks: Vec::new(),
            recently_finished_task_message: Vec::new(),
        }
    }

    fn pretty_print_tasks(&self) {
        // clear terminal
        print!("{esc}c", esc = 27 as char);

        let mut lines = vec![];

        lines.push(String::new());
        lines.push("Current tasks:".to_string());
        lines.push(String::new());
        for thread_id in &self.thread_ids_order {
            let stack = self.threads_stacks.get(thread_id).unwrap();
            let mut indent = 0;
            for task in stack {
                lines.push(format!(
                    "{}{} {}: {:.3}s",
                    " ".repeat(indent),
                    "`--",
                    task.name,
                    task.start.elapsed().as_secs_f32()
                ));
                indent += 1;
            }
        }

        lines.push(String::new());
        lines.push("Recently finished tasks (longer than 100ms):".to_string());
        lines.push(String::new());
        let max_task_name_length = self
            .recently_finished_long_tasks
            .iter()
            .map(|task| task.name.len())
            .max()
            .unwrap_or(0);

        for task in &self.recently_finished_long_tasks {
            let duration = elapsed(task.end.unwrap(),task.start);
            let formatted_task_name = format!("{:width$}", task.name, width = max_task_name_length);
            lines.push(format!(
                "- {:.3}s {}",
                duration.as_secs_f32(),
                formatted_task_name
            ));
        }

        let longest_line_length = lines.iter().map(|line| line.len()).max().unwrap_or(0);

        for i in 0..TERMINAL_HEIGHT {
            let padding = " ".repeat(longest_line_length);
            let line = lines.get(i).unwrap_or(&padding);
            let padding = " ".repeat(longest_line_length - line.len());
            let empty = String::new();
            let message_line = self.recently_finished_task_message.get(i).unwrap_or(&empty);
            println!("{}{}  ⁚  {}", line, padding, message_line);
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

    fn _end_task(&mut self, thread_id: ThreadId, end_message: Option<String>) -> Option<Task> {
        let stack = self._get_stack_or_create(thread_id);
        let maybe_task = stack.pop();
        if let Some(mut task) = maybe_task.clone() {
            task.end = Some(Instant::now());
            let full_name = stack
                .iter()
                .map(|t| t.name.clone())
                .collect::<Vec<_>>()
                .join(".");
            task.name = format!("{}.{}", full_name, task.name);

            if let Some(message) = end_message {
                let mut lines = vec![
                    "".to_string(),
                    format!(
                        "{} finished in {:.3}s with message:",
                        task.name,
                        (elapsed(task.end.unwrap(),task.start)).as_secs_f32()
                    ), 
                    "".to_string()
                ];
                let msg_lines = message
                    .split("\n")
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>();
                lines.extend(msg_lines);

                self.recently_finished_task_message.extend(lines);
                if self.recently_finished_task_message.len() > TERMINAL_HEIGHT {
                    self.recently_finished_task_message
                        .drain(0..self.recently_finished_task_message.len() - TERMINAL_HEIGHT);
                }
            }

            if task.start.elapsed() > Duration::from_millis(100) {
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

    fn listen(mut self, rx: Receiver<Messages>) {
        loop {
            if let Ok(message) = rx.recv() {
                match message.clone() {
                    Messages::Start => break,
                    _ => {}
                };
            }
        }

        let mut last_refresh = Instant::now();
        loop {
            if let Ok(message) = rx.recv_timeout(Duration::from_millis(500) - last_refresh.elapsed()) {
                match message.clone() {
                    Messages::Start => {}
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
                    Messages::EndTask { thread_id, message } => {
                        let _ = self._end_task(thread_id, message).unwrap();
                    }
                    Messages::Stop => {
                        self.pretty_print_tasks();
                        break;
                    },
                };
            }

            if last_refresh.elapsed() > Duration::from_millis(500) {
                self.pretty_print_tasks();
                last_refresh = Instant::now();
            }
        }
    }

    fn run() -> Sender<Messages> {
        let (tx, rx) = mpsc::channel::<Messages>();
        thread::spawn(move || {
            let gi = TM::new();
            gi.listen(rx);
        });

        tx.clone()
    }

    fn send(message: Messages) {
        let sender = SENDER_TO_GI.lock().unwrap();
        sender.send(message).unwrap();
    }

    pub fn start_monitoring() {
        Self::send(Messages::Start);
    }

    pub fn start<S: ToString>(name: S) {
        Self::send(Messages::StartTask {
            name: name.to_string(),
            thread_id: thread::current().id(),
        });
    }

    pub fn end() {
        Self::send(Messages::EndTask {
            thread_id: thread::current().id(),
            message: None,
        });
    }

    pub fn end_with_message<S: ToString>(message: S) {
        Self::send(Messages::EndTask {
            thread_id: thread::current().id(),
            message: Some(message.to_string()),
        });
    }

    pub fn stop_monitoring() {
        Self::send(Messages::Stop);
    }
}

pub fn elapsed(a: Instant, b: Instant) -> Duration {
    if a < b {
        Duration::from_secs(0)
    } else {
        a - b
    }
}