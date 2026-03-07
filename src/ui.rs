use console::Style;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

pub struct Ui {
    fancy: bool,
}

impl Ui {
    pub fn new(fancy: bool) -> Self {
        Self { fancy }
    }

    pub fn spinner(&self, message: &str) -> Spinner {
        if !self.fancy {
            eprintln!("{}", message);
            return Spinner { bar: None };
        }

        let bar = ProgressBar::new_spinner();
        bar.set_style(
            ProgressStyle::with_template("{spinner:.cyan} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        bar.set_message(message.to_string());
        bar.enable_steady_tick(Duration::from_millis(80));
        Spinner { bar: Some(bar) }
    }

    pub fn header(&self, text: &str) {
        if self.fancy {
            let style = Style::new().bold().cyan();
            eprintln!("\n{}", style.apply_to(text));
        } else {
            eprintln!("\n{}", text);
        }
    }

    pub fn success(&self, text: &str) {
        if self.fancy {
            let style = Style::new().green().bold();
            eprintln!("{} {}", style.apply_to("✓"), text);
        } else {
            eprintln!("{}", text);
        }
    }

    pub fn warn(&self, text: &str) {
        if self.fancy {
            let style = Style::new().yellow();
            eprintln!("{} {}", style.apply_to("⚠"), text);
        } else {
            eprintln!("WARNING: {}", text);
        }
    }

    pub fn error(&self, text: &str) {
        if self.fancy {
            let style = Style::new().red().bold();
            eprintln!("{} {}", style.apply_to("✗"), text);
        } else {
            eprintln!("ERROR: {}", text);
        }
    }

    pub fn detail(&self, label: &str, value: &str) {
        if self.fancy {
            let label_style = Style::new().dim();
            eprintln!("  {} {}", label_style.apply_to(format!("{}:", label)), value);
        } else {
            eprintln!("  {}: {}", label, value);
        }
    }

    pub fn divider(&self) {
        if self.fancy {
            let style = Style::new().dim();
            eprintln!("{}", style.apply_to("─".repeat(60)));
        } else {
            eprintln!("{}", "-".repeat(60));
        }
    }

    pub fn stats_block(&self, text: &str) {
        if self.fancy {
            let style = Style::new().dim();
            for line in text.lines() {
                eprintln!("{}", style.apply_to(line));
            }
        } else {
            eprintln!("{}", text);
        }
    }
}

pub struct Spinner {
    bar: Option<ProgressBar>,
}

impl Spinner {
    pub fn finish(&self, message: &str) {
        if let Some(bar) = &self.bar {
            bar.finish_with_message(message.to_string());
        }
    }

    pub fn finish_clear(&self) {
        if let Some(bar) = &self.bar {
            bar.finish_and_clear();
        }
    }

    pub fn set_message(&self, message: &str) {
        if let Some(bar) = &self.bar {
            bar.set_message(message.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_ui_does_not_panic() {
        let ui = Ui::new(false);
        ui.header("test header");
        ui.success("done");
        ui.warn("careful");
        ui.error("oops");
        ui.detail("key", "value");
        ui.divider();
        ui.stats_block("some stats");
    }

    #[test]
    fn fancy_ui_does_not_panic() {
        let ui = Ui::new(true);
        ui.header("test header");
        ui.success("done");
        ui.warn("careful");
        ui.error("oops");
        ui.detail("key", "value");
        ui.divider();
        ui.stats_block("some stats");
    }

    #[test]
    fn spinner_plain_mode() {
        let ui = Ui::new(false);
        let spinner = ui.spinner("loading...");
        spinner.finish("done");
    }

    #[test]
    fn spinner_fancy_mode() {
        let ui = Ui::new(true);
        let spinner = ui.spinner("loading...");
        spinner.set_message("still loading...");
        spinner.finish_clear();
    }
}
