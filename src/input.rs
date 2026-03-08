use std::io::BufRead;

use rustyline::config::EditMode;
use rustyline::error::ReadlineError;
use rustyline::history::DefaultHistory;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Completer, Editor, Helper, Highlighter, Hinter};

/// Validator that returns `Incomplete` until the user enters a blank line
/// (i.e. the buffer ends with `\n\n`), or `Valid` when the entire buffer is
/// empty/whitespace (user pressed Enter immediately to skip).
///
/// Also accepts input immediately when the last line is a `/command`
/// (e.g. `/done`, `/use 1`), so users don't need a trailing blank line
/// after commands.
#[derive(Completer, Helper, Highlighter, Hinter)]
pub struct MultilineHelper;

impl Validator for MultilineHelper {
    fn validate(&self, ctx: &mut ValidationContext<'_>) -> rustyline::Result<ValidationResult> {
        let input = ctx.input();
        if input.is_empty() {
            return Ok(ValidationResult::Valid(None));
        }
        // Double-newline (blank line) submits
        if input.ends_with("\n\n") || input.ends_with("\r\n\r\n") {
            return Ok(ValidationResult::Valid(None));
        }
        // A `/command` on the last line submits immediately
        let last_line = input.lines().next_back().unwrap_or("");
        if last_line.trim().starts_with('/') {
            return Ok(ValidationResult::Valid(None));
        }
        Ok(ValidationResult::Incomplete)
    }
}

fn vi_config() -> rustyline::Config {
    rustyline::Config::builder()
        .edit_mode(EditMode::Vi)
        .auto_add_history(true)
        .build()
}

pub struct InputEditor {
    single: Editor<(), DefaultHistory>,
    multi: Editor<MultilineHelper, DefaultHistory>,
}

impl InputEditor {
    pub fn new() -> anyhow::Result<Self> {
        let single = Editor::<(), DefaultHistory>::with_config(vi_config())?;
        let mut multi = Editor::<MultilineHelper, DefaultHistory>::with_config(vi_config())?;
        multi.set_helper(Some(MultilineHelper));
        Ok(Self { single, multi })
    }

    /// Single-line prompt. Returns `Ok(None)` on empty/EOF/Ctrl-C.
    pub fn read_line(&mut self, prompt: &str) -> anyhow::Result<Option<String>> {
        match self.single.readline(prompt) {
            Ok(line) => {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(trimmed))
                }
            }
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Loops until non-empty input. Returns error on EOF/Ctrl-C.
    pub fn read_required_line(&mut self, prompt: &str) -> anyhow::Result<String> {
        loop {
            match self.single.readline(prompt) {
                Ok(line) => {
                    let trimmed = line.trim().to_string();
                    if !trimmed.is_empty() {
                        return Ok(trimmed);
                    }
                    println!("Value cannot be empty.");
                }
                Err(ReadlineError::Interrupted) => {
                    return Err(anyhow::anyhow!("input cancelled"));
                }
                Err(ReadlineError::Eof) => {
                    return Err(anyhow::anyhow!("stdin closed while reading input"));
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    /// Multiline prompt (blank line to finish). Returns `Ok(None)` on empty/EOF/Ctrl-C.
    pub fn read_multiline(&mut self, prompt: &str) -> anyhow::Result<Option<String>> {
        match self.multi.readline(prompt) {
            Ok(text) => {
                let trimmed = text.trim().to_string();
                if trimmed.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(trimmed))
                }
            }
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

/// Read multiline input from a `BufRead` source (for testing).
pub fn read_multiline_from<R: BufRead>(handle: &mut R) -> anyhow::Result<String> {
    let mut input = String::new();
    let mut line = String::new();

    loop {
        line.clear();
        let bytes = handle.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
        if trimmed.is_empty() {
            break;
        }
        input.push_str(trimmed);
        input.push('\n');
    }

    Ok(input.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn read_multiline_from_stops_on_blank_line() {
        let input = "line1\nline2\n\nline3\n";
        let mut reader = Cursor::new(input);
        let output = read_multiline_from(&mut reader).expect("read");
        assert_eq!(output, "line1\nline2");
    }

    #[test]
    fn read_multiline_from_reads_until_eof() {
        let input = "line1\r\nline2";
        let mut reader = Cursor::new(input);
        let output = read_multiline_from(&mut reader).expect("read");
        assert_eq!(output, "line1\nline2");
    }
}
