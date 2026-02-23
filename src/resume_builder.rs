use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use anyhow::{anyhow, Context};
use rig::providers::openai::responses_api::ResponsesCompletionModel;
use crate::{kb, prompts};
use crate::kb::UserProfile;
use crate::llm::prompt_text;

const MAX_LATEX_FIXES: usize = 3;

struct LatexCompileResult {
    log: String,
    has_warning: bool,
    has_error: bool,
    success: bool,
}

fn compile_latex(path: PathBuf) -> anyhow::Result<LatexCompileResult> {
    let output = Command::new("pdflatex")
        .arg("-interaction=nonstopmode")
        .arg("-halt-on-error")
        .arg("-file-line-error")
        .arg(path)
        .output()
        .context("failed to run pdflatex")?;

    let mut log = String::new();
    log.push_str(&String::from_utf8_lossy(&output.stdout));
    if !output.stderr.is_empty() {
        if !log.ends_with('\n') {
            log.push('\n');
        }
        log.push_str(&String::from_utf8_lossy(&output.stderr));
    }

    let has_warning = log.contains("LaTeX Warning");
    let has_error = !output.status.success()
        || log.lines().any(|line| line.starts_with('!'))
        || log.contains("Emergency stop")
        || log.contains("Undefined control sequence");

    Ok(LatexCompileResult {
        log,
        has_warning,
        has_error,
        success: output.status.success(),
    })
}



fn format_user_profile(profile: &UserProfile) -> String {
    let mut out = Vec::new();
    out.push(format!("Name: {}", profile.name));
    out.push(format!("Location: {}", profile.location));
    out.push(format!("Email: {}", profile.email));
    out.push(format!("Phone: {}", profile.phone));

    if profile.links.is_empty() {
        out.push("Links: None.".to_string());
    } else {
        out.push("Links:".to_string());
        for link in &profile.links {
            out.push(format!("- {}: {}", link.label, link.url));
        }
    }

    if profile.education.is_empty() {
        out.push("Education: None.".to_string());
    } else {
        out.push("Education:".to_string());
        for edu in &profile.education {
            out.push(format!("- {} (Graduation: {})", edu.degree, edu.graduation_date));
        }
    }

    if profile.jobs.is_empty() {
        out.push("Job History: None.".to_string());
    } else {
        out.push("Job History:".to_string());
        for job in &profile.jobs {
            out.push(format!(
                "- {} — {}, {} ({} to {})",
                job.company, job.title, job.location, job.start_date, job.end_date
            ));
        }
    }

    out.join("\n")
}

fn tail_lines(input: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = input.lines().collect();
    if lines.len() <= max_lines {
        return input.to_string();
    }
    lines[lines.len().saturating_sub(max_lines)..].join("\n")
}

async fn fix_and_save_latex(
    out_dir: &Path,
    resume_latex: &mut String,
    completion_model: &ResponsesCompletionModel,
) -> anyhow::Result<()>{

    let output_tex = out_dir.join("resume.tex");

    for attempt in 0..=MAX_LATEX_FIXES {
        fs::write(&output_tex, resume_latex.as_str())
            .with_context(|| format!("failed to write resume to {}", output_tex.to_str().unwrap()))?;

        let compile = compile_latex(out_dir.join("resume.pdf"))?;
        if compile.success && !compile.has_warning && !compile.has_error {
            println!("Wrote PDF resume to generated_resume.pdf");
            break;
        }

        fs::write("generated_resume.compile.log", &compile.log).ok();

        if attempt == MAX_LATEX_FIXES {
            if compile.has_error {
                return Err(anyhow!(
                    "pdflatex failed after {} attempts. See generated_resume.compile.log",
                    attempt + 1
                ));
            }
            println!(
                "LaTeX compiled with warnings after {} attempts. See generated_resume.compile.log",
                attempt + 1
            );
            println!("Wrote PDF resume to generated_resume.pdf");
            break;
        }

        let issue = match (compile.has_error, compile.has_warning) {
            (true, true) => "errors and warnings",
            (true, false) => "errors",
            (false, true) => "warnings",
            (false, false) => "issues",
        };
        println!(
            "LaTeX compilation reported {}. Attempting auto-fix ({}/{})...",
            issue,
            attempt + 1,
            MAX_LATEX_FIXES + 1
        );

        let compile_tail = tail_lines(&compile.log, 200);
        let fix_prompt = format!(
            "LATEX DOCUMENT:\n{}\n\nCOMPILER OUTPUT:\n{}\n",
            resume_latex, compile_tail
        );
        *resume_latex =
            prompt_text(completion_model, prompts::RESUME_FIX_PREAMBLE, &fix_prompt).await?;
    }

    Ok(())
}
pub async fn build_resume(job_text: &String,
                          user_profile: UserProfile,
                          out_dir: &Path,
                          completion_model: &ResponsesCompletionModel) -> anyhow::Result<()> {
    let kb_docs = kb::list_story_documents()?;
    let kb_context = if kb_docs.is_empty() {
        "None.".to_string()
    } else {
        kb_docs
            .iter()
            .map(|doc| format!("- {doc}"))
            .collect::<Vec<_>>()
            .join("\n")
    };

    // let starter_resume = args
    //     .get(2)
    //     .map(|path| {
    //         fs::read_to_string(path)
    //             .with_context(|| format!("failed to read file: {}", path))
    //     })
    //     .transpose()?;
    //
    // let starter_resume = starter_resume
    //     .map(|text| text.trim().to_string())
    //     .filter(|text| !text.is_empty())
    //     .unwrap_or_else(|| "None provided.".to_string());
    //

    let user_profile_context = format_user_profile(&user_profile);
    let resume_prompt = format!(
        "JOB DESCRIPTION:\n{}\n\nKNOWLEDGE BASE:\n{}\n\nUSER PROFILE:\n{}\n\nTEMPLATE:\n{}\n",
        job_text, kb_context, user_profile_context, prompts::RESUME_TEMPLATE_LATEX
    );



    let mut resume_latex = prompt_text(completion_model,
                                       prompts::RESUME_BUILD_PREAMBLE,
                                       &resume_prompt).await?;


    if !out_dir.exists() {
        fs::create_dir_all(out_dir)
            .with_context(|| format!("failed to create output directory: {}", out_dir.display()))?;
    }

    fix_and_save_latex(out_dir, &mut resume_latex, completion_model).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kb::{EducationEntry, JobEntry, ProfileLink, UserProfile};

    #[test]
    fn tail_lines_returns_input_when_under_limit() {
        let input = "line1\nline2\nline3";
        let output = tail_lines(input, 5);
        assert_eq!(output, input);
    }

    #[test]
    fn tail_lines_returns_last_lines() {
        let input = "line1\nline2\nline3\nline4";
        let output = tail_lines(input, 2);
        assert_eq!(output, "line3\nline4");
    }

    #[test]
    fn format_user_profile_with_empty_sections() {
        let profile = UserProfile {
            name: "Ada".to_string(),
            location: "NYC".to_string(),
            email: "ada@example.com".to_string(),
            phone: "555-1234".to_string(),
            links: Vec::new(),
            education: Vec::new(),
            jobs: Vec::new(),
        };
        let formatted = format_user_profile(&profile);
        let expected = "\
Name: Ada
Location: NYC
Email: ada@example.com
Phone: 555-1234
Links: None.
Education: None.
Job History: None.";
        assert_eq!(formatted, expected);
    }

    #[test]
    fn format_user_profile_with_entries() {
        let profile = UserProfile {
            name: "Ada".to_string(),
            location: "NYC".to_string(),
            email: "ada@example.com".to_string(),
            phone: "555-1234".to_string(),
            links: vec![ProfileLink {
                label: "GitHub".to_string(),
                url: "https://github.com/ada".to_string(),
            }],
            education: vec![EducationEntry {
                degree: "BS CS".to_string(),
                graduation_date: "2020".to_string(),
            }],
            jobs: vec![JobEntry {
                company: "Acme".to_string(),
                title: "Engineer".to_string(),
                location: "Remote".to_string(),
                start_date: "2020".to_string(),
                end_date: "2022".to_string(),
            }],
        };
        let formatted = format_user_profile(&profile);
        let expected = "\
Name: Ada
Location: NYC
Email: ada@example.com
Phone: 555-1234
Links:
- GitHub: https://github.com/ada
Education:
- BS CS (Graduation: 2020)
Job History:
- Acme — Engineer, Remote (2020 to 2022)";
        assert_eq!(formatted, expected);
    }
}
