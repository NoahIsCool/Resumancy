use std::fs;
use std::path::Path;
use std::process::Command;
use anyhow::{anyhow, Context};
use rig::completion::{CompletionModel, Usage};
use crate::{eval, kb, prompts};
use crate::kb::UserProfile;
use crate::hiring_manager::SkillFocusList;
use crate::llm::{prompt_text_with_temperature, prompt_text_streaming, strip_code_fences, combine_usage, CacheConfig, Provider};

const MAX_LATEX_FIXES: usize = 3;
const TOP_STORIES_PER_SKILL: usize = 50;

const RESUME_TEMPLATE: &str = include_str!("../data/resume_template.tex");
const COVER_LETTER_TEMPLATE: &str = include_str!("../data/cover_letter_template.tex");

/// Load a LaTeX template: check `data_dir()/filename` for a user override,
/// otherwise fall back to the embedded default.
fn load_template(filename: &str, embedded: &str) -> String {
    if let Ok(data_dir) = crate::paths::data_dir() {
        let override_path = data_dir.join(filename);
        if let Ok(contents) = fs::read_to_string(&override_path) {
            return contents;
        }
    }
    embedded.to_string()
}

struct LatexCompileResult {
    log: String,
    has_warning: bool,
    has_error: bool,
    success: bool,
}

fn compile_latex(input_path: &Path, out_dir: &Path) -> anyhow::Result<LatexCompileResult> {
    let output = Command::new("pdflatex")
        .arg("-interaction=nonstopmode")
        .arg("-halt-on-error")
        .arg("-file-line-error")
        .arg("-output-directory")
        .arg(out_dir)
        .arg(input_path)
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

fn format_skill_priorities(skill_assessment: &SkillFocusList) -> String {
    let mut out = Vec::new();
    out.push(format!("Overall: {}", skill_assessment.summary));
    for (idx, skill) in skill_assessment.skills.iter().enumerate() {
        out.push(format!(
            "{}. {} (need: {}/9, suitability: {}/9) — {}",
            idx + 1, skill.title, skill.need, skill.suitability, skill.justification
        ));
    }
    out.join("\n")
}

async fn build_matched_stories_context(
    skill_assessment: &SkillFocusList,
    embed_model: &impl rig::embeddings::EmbeddingModel,
) -> anyhow::Result<String> {
    let mut seen_texts = std::collections::HashSet::new();
    let mut sections = Vec::new();

    for skill in &skill_assessment.skills {
        let query = format!("{}: {}", skill.title, skill.skill_description);
        let stories = kb::retrieve_relevant_stories(&query, TOP_STORIES_PER_SKILL, embed_model).await?;

        if stories.is_empty() {
            continue;
        }

        let mut skill_stories = Vec::new();
        for story in &stories {
            let doc = kb::story_document(&story.company, &story.year, &story.text);
            if seen_texts.insert(doc.clone()) {
                skill_stories.push(format!("  - {}", doc));
            }
        }

        if !skill_stories.is_empty() {
            sections.push(format!("{}:\n{}", skill.title, skill_stories.join("\n")));
        }
    }

    if sections.is_empty() {
        Ok("None.".to_string())
    } else {
        Ok(sections.join("\n\n"))
    }
}

fn tail_lines(input: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = input.lines().collect();
    if lines.len() <= max_lines {
        return input.to_string();
    }
    lines[lines.len().saturating_sub(max_lines)..].join("\n")
}

async fn fix_and_save_latex<M: CompletionModel + Clone>(
    out_dir: &Path,
    tex_filename: &str,
    resume_latex: &mut String,
    completion_model: &M,
    cache: Option<&CacheConfig>,
) -> anyhow::Result<Usage>{
    let mut total_usage = Usage::default();

    let output_tex = out_dir.join(format!("{}.tex", tex_filename));
    let output_pdf = out_dir.join(format!("{}.pdf", tex_filename));
    let output_log = out_dir.join(format!("{}.compile.log", tex_filename));

    for attempt in 0..=MAX_LATEX_FIXES {
        fs::write(&output_tex, resume_latex.as_str())
            .with_context(|| format!("failed to write resume to {}", output_tex.to_str().unwrap()))?;

        let compile = compile_latex(&output_tex, out_dir)?;
        if compile.success && !compile.has_warning && !compile.has_error {
            println!("Wrote PDF resume to {}", output_pdf.display());
            break;
        }

        fs::write(&output_log, &compile.log).ok();

        if attempt == MAX_LATEX_FIXES {
            if compile.has_error {
                return Err(anyhow!(
                    "pdflatex failed after {} attempts. See {}",
                    attempt + 1,
                    output_log.display()
                ));
            }
            println!(
                "LaTeX compiled with warnings after {} attempts. See {}",
                attempt + 1,
                output_log.display()
            );
            println!("Wrote PDF resume to {}", output_pdf.display());
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
        let fix_result = prompt_text_with_temperature(completion_model, prompts::RESUME_FIX_PREAMBLE, &fix_prompt, 0.4, cache).await?;
        total_usage = combine_usage(total_usage, fix_result.usage);
        *resume_latex = strip_code_fences(&fix_result.value).to_string();
    }

    Ok(total_usage)
}

#[tracing::instrument(skip_all)]
pub async fn build_resume<M: CompletionModel + Clone>(
    job_text: &String,
    user_profile: UserProfile,
    skill_assessment: &SkillFocusList,
    out_dir: &Path,
    tex_filename: &str,
    completion_model: &M,
    embed_model: &impl rig::embeddings::EmbeddingModel,
    provider: Provider,
    run_eval: bool,
    stream: bool,
    cache: Option<&CacheConfig>,
) -> anyhow::Result<(Option<eval::ResumeEvaluation>, Usage)> {
    let mut total_usage = Usage::default();

    let skill_priorities = format_skill_priorities(skill_assessment);
    let matched_stories = build_matched_stories_context(skill_assessment, embed_model).await?;
    let user_profile_context = format_user_profile(&user_profile);

    let template = load_template("resume_template.tex", RESUME_TEMPLATE);

    let resume_prompt = format!(
        "JOB DESCRIPTION:\n{}\n\nSKILL PRIORITIES:\n{}\n\nMATCHED STORIES:\n{}\n\nUSER PROFILE:\n{}\n\nSTARTER RESUME:\n{}\n\nTEMPLATE:\n{}\n",
        job_text,
        skill_priorities,
        matched_stories,
        user_profile_context,
        "None provided.",
        template
    );

    let raw = if stream {
        prompt_text_streaming(completion_model, prompts::RESUME_BUILD_PREAMBLE, &resume_prompt, 0.4, cache).await?
    } else {
        let result = prompt_text_with_temperature(completion_model, prompts::RESUME_BUILD_PREAMBLE, &resume_prompt, 0.4, cache).await?;
        total_usage = combine_usage(total_usage, result.usage);
        result.value
    };
    let mut resume_latex = strip_code_fences(&raw).to_string();

    let evaluation = if run_eval {
        let resume_prompt_clone = resume_prompt.clone();
        let cache_clone = cache.cloned();
        let (final_latex, eval_result) = eval::eval_loop(
            &resume_latex,
            job_text,
            completion_model,
            provider,
            None,
            cache,
            |eval_feedback| {
                let suggestions = eval_feedback.suggestions.join("\n- ");
                let weaknesses = eval_feedback.weaknesses.join("\n- ");
                let regen_prompt = format!(
                    "{}\n\nPREVIOUS EVALUATION FEEDBACK:\nWeaknesses:\n- {}\n\nSuggestions:\n- {}\n\nPlease regenerate an improved resume.",
                    resume_prompt_clone, weaknesses, suggestions
                );
                let model = completion_model.clone();
                let cache_inner = cache_clone.clone();
                async move {
                    let raw = prompt_text_with_temperature(&model, prompts::RESUME_REGENERATE_PREAMBLE, &regen_prompt, 0.4, cache_inner.as_ref()).await?.value;
                    Ok(strip_code_fences(&raw).to_string())
                }
            },
        ).await?;
        resume_latex = final_latex;
        Some(eval_result)
    } else {
        None
    };

    if !out_dir.exists() {
        fs::create_dir_all(out_dir)
            .with_context(|| format!("failed to create output directory: {}", out_dir.display()))?;
    }

    let fix_usage = fix_and_save_latex(out_dir, tex_filename, &mut resume_latex, completion_model, cache).await?;
    total_usage = combine_usage(total_usage, fix_usage);

    Ok((evaluation, total_usage))
}

#[tracing::instrument(skip_all)]
pub async fn build_cover_letter<M: CompletionModel + Clone>(
    job_text: &String,
    user_profile: &UserProfile,
    skill_assessment: &SkillFocusList,
    out_dir: &Path,
    tex_filename: &str,
    completion_model: &M,
    embed_model: &impl rig::embeddings::EmbeddingModel,
    stream: bool,
    cache: Option<&CacheConfig>,
) -> anyhow::Result<Usage> {
    let mut total_usage = Usage::default();

    let skill_priorities = format_skill_priorities(skill_assessment);
    let matched_stories = build_matched_stories_context(skill_assessment, embed_model).await?;
    let user_profile_context = format_user_profile(user_profile);

    let template = load_template("cover_letter_template.tex", COVER_LETTER_TEMPLATE);

    let cover_letter_prompt = format!(
        "JOB DESCRIPTION:\n{}\n\nSKILL PRIORITIES:\n{}\n\nMATCHED STORIES:\n{}\n\nUSER PROFILE:\n{}\n\nTEMPLATE:\n{}\n",
        job_text,
        skill_priorities,
        matched_stories,
        user_profile_context,
        template,
    );

    let raw = if stream {
        prompt_text_streaming(completion_model, prompts::COVER_LETTER_PREAMBLE, &cover_letter_prompt, 0.4, cache).await?
    } else {
        let result = prompt_text_with_temperature(completion_model, prompts::COVER_LETTER_PREAMBLE, &cover_letter_prompt, 0.4, cache).await?;
        total_usage = combine_usage(total_usage, result.usage);
        result.value
    };
    let mut cover_latex = strip_code_fences(&raw).to_string();

    if !out_dir.exists() {
        fs::create_dir_all(out_dir)
            .with_context(|| format!("failed to create output directory: {}", out_dir.display()))?;
    }

    let fix_usage = fix_and_save_latex(out_dir, tex_filename, &mut cover_latex, completion_model, cache).await?;
    total_usage = combine_usage(total_usage, fix_usage);

    Ok(total_usage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kb::{EducationEntry, JobEntry, ProfileLink, UserProfile};
    use crate::hiring_manager::{SkillFocusList, SkillNeed};

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

    #[test]
    fn format_skill_priorities_formats_correctly() {
        let assessment = SkillFocusList {
            summary: "Strong fit".to_string(),
            skills: vec![
                SkillNeed {
                    title: "Rust".to_string(),
                    description: "Systems programming".to_string(),
                    need: 9,
                    suitability: 7,
                    skill_description: "Rust language".to_string(),
                    justification: "Strong experience".to_string(),
                },
                SkillNeed {
                    title: "Python".to_string(),
                    description: "Scripting".to_string(),
                    need: 5,
                    suitability: 8,
                    skill_description: "Python scripting".to_string(),
                    justification: "Extensive use".to_string(),
                },
            ],
        };
        let formatted = format_skill_priorities(&assessment);
        assert!(formatted.contains("Overall: Strong fit"));
        assert!(formatted.contains("1. Rust (need: 9/9, suitability: 7/9)"));
        assert!(formatted.contains("2. Python (need: 5/9, suitability: 8/9)"));
    }
}
