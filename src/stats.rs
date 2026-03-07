use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct LlmCall {
    pub label: String,
    pub model: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub duration: Duration,
}

#[derive(Debug, Clone, Default)]
struct Inner {
    calls: Vec<LlmCall>,
}

#[derive(Debug, Clone)]
pub struct StatsCollector {
    inner: Arc<Mutex<Inner>>,
    enabled: bool,
}

impl StatsCollector {
    pub fn new(enabled: bool) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner::default())),
            enabled,
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn record(&self, call: LlmCall) {
        if !self.enabled {
            return;
        }
        let mut inner = self.inner.lock().expect("stats lock");
        inner.calls.push(call);
    }

    pub fn record_with_usage(&self, label: &str, model: &str, usage: rig::completion::Usage, duration: Duration) {
        self.record(LlmCall {
            label: label.to_string(),
            model: model.to_string(),
            prompt_tokens: usage.input_tokens,
            completion_tokens: usage.output_tokens,
            duration,
        });
    }

    pub fn start_timer(&self) -> CallTimer {
        CallTimer {
            start: Instant::now(),
        }
    }

    pub fn summary(&self) -> String {
        let inner = self.inner.lock().expect("stats lock");
        if inner.calls.is_empty() {
            return "No LLM calls recorded.".to_string();
        }

        let total_calls = inner.calls.len();
        let total_prompt: u64 = inner.calls.iter().map(|c| c.prompt_tokens).sum();
        let total_completion: u64 = inner.calls.iter().map(|c| c.completion_tokens).sum();
        let total_tokens = total_prompt + total_completion;
        let total_duration: Duration = inner.calls.iter().map(|c| c.duration).sum();

        let mut lines = Vec::new();
        lines.push(format!("LLM Usage Summary"));
        lines.push(format!("{}", "-".repeat(60)));

        for (i, call) in inner.calls.iter().enumerate() {
            let tokens = call.prompt_tokens + call.completion_tokens;
            lines.push(format!(
                "  {:2}. {:<30} {:>6} tok  {:.1}s  ({})",
                i + 1,
                truncate(&call.label, 30),
                tokens,
                call.duration.as_secs_f64(),
                call.model,
            ));
        }

        lines.push(format!("{}", "-".repeat(60)));
        lines.push(format!(
            "  Total: {} calls, {} tokens ({}p + {}c), {:.1}s",
            total_calls,
            total_tokens,
            total_prompt,
            total_completion,
            total_duration.as_secs_f64(),
        ));

        lines.join("\n")
    }
}

pub struct CallTimer {
    start: Instant,
}

impl CallTimer {
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_collector_ignores_records() {
        let stats = StatsCollector::new(false);
        stats.record(LlmCall {
            label: "test".into(),
            model: "gpt-4o".into(),
            prompt_tokens: 100,
            completion_tokens: 50,
            duration: Duration::from_millis(500),
        });
        assert_eq!(stats.summary(), "No LLM calls recorded.");
    }

    #[test]
    fn enabled_collector_tracks_calls() {
        let stats = StatsCollector::new(true);
        stats.record(LlmCall {
            label: "job_needs".into(),
            model: "claude-sonnet".into(),
            prompt_tokens: 200,
            completion_tokens: 100,
            duration: Duration::from_millis(1500),
        });
        stats.record(LlmCall {
            label: "evaluation".into(),
            model: "claude-sonnet".into(),
            prompt_tokens: 300,
            completion_tokens: 150,
            duration: Duration::from_millis(2000),
        });
        let summary = stats.summary();
        assert!(summary.contains("2 calls"));
        assert!(summary.contains("750 tokens"));
        assert!(summary.contains("job_needs"));
        assert!(summary.contains("evaluation"));
    }

    #[test]
    fn timer_measures_elapsed() {
        let stats = StatsCollector::new(true);
        let timer = stats.start_timer();
        std::thread::sleep(Duration::from_millis(10));
        assert!(timer.elapsed() >= Duration::from_millis(5));
    }

    #[test]
    fn truncate_short_string() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_long_string() {
        let result = truncate("a very long label name here", 15);
        assert!(result.len() <= 15);
        assert!(result.ends_with("..."));
    }
}
