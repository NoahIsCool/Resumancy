mod build_master;

use std::{
    env,
    fs,
    io::{self, BufRead, Write},
    process::Command,
};
use std::collections::HashMap;
use anyhow::{anyhow, Context, Result};
use pipelines::kb::{EducationEntry, JobEntry, ProfileLink, StorySeed, UserProfile};
use pipelines::llm::{
    openai_client_from_env, prompt_structured, prompt_text, EMBEDDING_MODEL_NAME, MODEL_NAME,
};
use rig::client::{CompletionClient, EmbeddingsClient};
use rig::completion::CompletionModel;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const MAX_SKILL_TURNS: usize = 4;
const MAX_LATEX_FIXES: usize = 3;
const JOB_POST_PREAMBLE: &str = r#"Role: You are a hiring manager responsible for digitizing job roles. Given a job posting, identify the attributes or skills required by the posting. Identify every key word and industry skill mentioned. Be sure to "Read between the lines" to include skills that are implied, but not explicitly mentioned. For example, A role as an "AI Researger" would likely faveor a graduate degree in AI, and even though it is not explicitly mentioned, it would likely prefer experience pubishing research papers.

For each skill, return:
- title: short label of the skill
- description: 1-2 sentences describing what the skill entails in this role
- need: ranking from 0-9 of how nessasery the skill is to the job posting. 0 is not needed at all, 9 is a skill absolutely required, and anything less than 5 is simply nice to have.
- skill_description: short phrase (3-8 words) describing the skill area
"#;

const SKILL_PLAN_PREAMBLE: &str = r#"You are a hiring manager. Compare an explicit set of parsed needed skills against a knowledge base of applicant skills to identify how well the candidate satisfies each requirement.

Rules:
- Use only the provided JOB NEEDS list as the authoritative set of skills. Do not add, remove, or rename skills.
- Copy title, need, description, and skill_description verbatim from JOB NEEDS.
- Provide suitability and justification based only on the retrieved evidence.
- If there is no evidence, set suitability low and explain the gap concisely.
- Preserve the same ordering as JOB NEEDS.

Give a summary of the condidate's sutiblity for the role. Then, for each skill, return a struct with the following fields:
- title (copied verbatim from JOB NEEDS)
- description (copied verbatim from JOB NEEDS)
- need (copied verbatim from JOB NEEDS)
- skill_description (copied verbatim from JOB NEEDS)
- suitability (0-9)
- justification (brief, evidence-based)
"#;

const STORY_ASSESS_PREAMBLE: &str = r#"You are a resume coach.
Given a target skill and the user's responses, decide the next action.
Rules:
- Always extract the story fields: company, year, text. Use empty strings for missing fields.
- Year should be a 4-digit year or "unknown" if unavailable.
- If the user states they have no direct experience, set action to "ask_adjacent" and provide one concise adjacent question.
- If any required fields are missing, set action to "ask_followup" and ask one concise follow-up question.
- If all required fields are present, set action to "save_story".
- missing_fields should list any missing required fields.
- If a question is not needed, set its field to an empty string."#;

const RESUME_BUILD_PREAMBLE: &str = r#"You are a resume writer. Build a tailored resume in LaTeX using the provided template.
Rules:
- Use only facts from JOB DESCRIPTION, KNOWLEDGE BASE, USER PROFILE, and STARTER RESUME.
- Do not invent employers, titles, dates, degrees, or metrics.
- Prefer KNOWLEDGE BASE for experience details; use USER PROFILE for biographical/contact/education details; use STARTER RESUME for any missing details if present.
- If a detail is missing, omit it rather than guessing.
- Focus on role fit from the job description.
- Keep bullet points concise and impact-focused.
- Output only valid LaTeX; no commentary or markdown."#;

const RESUME_FIX_PREAMBLE: &str = r#"You are a LaTeX build fixer.
Given a LaTeX document and compiler output, return a corrected full LaTeX document that compiles cleanly.
Rules:
- Preserve the original content and meaning; fix only syntax, escaping, and formatting issues.
- Prefer minimal edits.
- Output only LaTeX; no commentary or markdown."#;

const RESUME_TEMPLATE_LATEX: &str = r#"\documentclass[11pt]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries}{}{0em}{}[\titlerule]

\begin{document}
\begin{center}
{\LARGE \textbf{<NAME>}}\\
<LOCATION> \textbar <EMAIL> \textbar <PHONE> \textbar \href{<LINKEDIN_URL>}{LinkedIn}
\end{center}

\section*{Summary}
<SUMMARY>

\section*{Experience}
% Repeat this block per role.
\textbf{<COMPANY>} --- <TITLE> \hfill <LOCATION>\\
\textit{<DATES>}\\
\begin{itemize}[leftmargin=*]
\item <IMPACT_BULLET>
\item <IMPACT_BULLET>
\end{itemize}

\section*{Projects}
% Repeat this block per project if applicable.
\textbf{<PROJECT_NAME>} --- <ROLE_OR_CONTEXT> \hfill <DATES>\\
\begin{itemize}[leftmargin=*]
\item <IMPACT_BULLET>
\end{itemize}

\section*{Education}
\textbf{<SCHOOL>} --- <DEGREE> \hfill <DATES>\\
<DETAILS>

\section*{Skills}
\textbf{Languages:} <LANGUAGES>\\
\textbf{Tools:} <TOOLS>\\
\textbf{Domains:} <DOMAINS>

\end{document}"#;

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
struct JobNeeds {
    skills: Vec<Need>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
struct Need {
    title: String, // title of the skill
    description: String, // short description of the skill area.
    need: u8, //ranking from 0-9 of how necessary the skill is
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
struct SkillFocusList {
    summary: String,
    skills: Vec<SkillNeed>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
struct SkillNeed {
    title: String, // title of the skill
    description: String,
    need: u8, //ranking from 0-9 of how necessary the skill is
    suitability: u8, // a ranking from 0-9 of how well the candidate satisfies requirement
    skill_description: String, // short description of the skill area.
    justification: String, // a short justification of the candidate's ranking in this skill area.
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[serde(rename_all = "snake_case")]
enum NextAction {
    SaveStory,
    AskFollowup,
    AskAdjacent,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
struct ParsedStory {
    company: String,
    year: String,
    text: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
struct StoryAssessment {
    action: NextAction,
    missing_fields: Vec<String>,
    followup_question: String,
    adjacent_question: String,
    parsed_story: ParsedStory,
}

struct LatexCompileResult {
    log: String,
    has_warning: bool,
    has_error: bool,
    success: bool,
}

fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn read_required_line(prompt: &str) -> Result<String> {
    loop {
        print!("{prompt}");
        io::stdout().flush().ok();
        let mut line = String::new();
        let bytes = io::stdin().read_line(&mut line)?;
        if bytes == 0 {
            return Err(anyhow!("stdin closed while reading input"));
        }
        let value = line.trim_end_matches(&['\r', '\n'][..]).to_string();
        if !value.trim().is_empty() {
            return Ok(value);
        }
        println!("Value cannot be empty.");
    }
}

fn read_optional_line(prompt: &str) -> Result<Option<String>> {
    print!("{prompt}");
    io::stdout().flush().ok();
    let mut line = String::new();
    let bytes = io::stdin().read_line(&mut line)?;
    if bytes == 0 {
        return Ok(None);
    }
    let value = line.trim_end_matches(&['\r', '\n'][..]).to_string();
    if value.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(value))
}

fn collect_user_profile() -> Result<UserProfile> {
    println!("\n=== User profile ===");
    let name = read_required_line("Full name: ")?;
    let location = read_required_line("Location: ")?;
    let email = read_required_line("Email: ")?;
    let phone = read_required_line("Phone number: ")?;

    println!("\n--- Relevant links (e.g., GitHub, LinkedIn) ---");
    let mut links = Vec::new();
    loop {
        let label = read_optional_line("Link label (blank to finish): ")?;
        let Some(label) = label else { break };
        let url = read_required_line(&format!("URL for {}: ", label))?;
        links.push(ProfileLink { label, url });
    }

    println!("\n--- Education ---");
    let mut education = Vec::new();
    loop {
        let degree = read_optional_line("Degree title (blank to finish): ")?;
        let Some(degree) = degree else { break };
        let graduation_date = read_required_line("Graduation date: ")?;
        education.push(EducationEntry {
            degree,
            graduation_date,
        });
    }

    println!("\n--- Job history ---");
    let mut jobs = Vec::new();
    loop {
        let company = read_optional_line("Company name (blank to finish): ")?;
        let Some(company) = company else { break };
        let title = read_required_line("Job title: ")?;
        let job_location = read_required_line("Job location: ")?;
        let start_date = read_required_line("Start date: ")?;
        let end_date = read_required_line("End date (or Present): ")?;
        jobs.push(JobEntry {
            company,
            title,
            location: job_location,
            start_date,
            end_date,
        });
    }

    Ok(UserProfile {
        name,
        location,
        email,
        phone,
        links,
        education,
        jobs,
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

fn compile_latex(path: &str) -> Result<LatexCompileResult> {
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

fn missing_fields(parsed: &ParsedStory) -> Vec<String> {
    let mut missing = Vec::new();
    if parsed.company.trim().is_empty() {
        missing.push("company".to_string());
    }
    if parsed.year.trim().is_empty() {
        missing.push("year".to_string());
    }
    if parsed.text.trim().is_empty() {
        missing.push("text".to_string());
    }
    missing
}

fn read_multiline() -> Result<String> {
    let mut input = String::new();
    let mut line = String::new();
    let stdin = io::stdin();
    let mut handle = stdin.lock();

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

fn followup_fallback_question(missing_fields: &[String]) -> String {
    let mut parts = Vec::new();
    if !missing_fields.is_empty() {
        parts.push(format!("missing fields: {}", missing_fields.join(", ")));
    }
    if parts.is_empty() {
        "Could you add a bit more detail to your story?".to_string()
    } else {
        format!("Could you add details for {}?", parts.join(" | "))
    }
}

fn adjacent_fallback_question(skill: &SkillNeed) -> String {
    format!(
        "If you don't have direct experience with {}, what adjacent experience could you share?",
        skill.title
    )
}

async fn collect_story_for_skill<M>(model: &M, skill: &SkillNeed) -> Result<Option<ParsedStory>>
where
    M: CompletionModel + Clone,
{
    println!("Target skill: {}", skill.title);
    println!("Skill description: {}", skill.skill_description);
    println!(
        "Share one story that demonstrates this skill. Include:\n\
- company\n- year (or \"unknown\")\n- story text (1-3 sentences)\n\
Finish with an empty line."
    );
    io::stdout().flush().ok();

    let mut combined_input = read_multiline()?;
    if combined_input.trim().is_empty() {
        return Ok(None);
    }

    for turn in 0..MAX_SKILL_TURNS {
        let assessment_prompt = format!(
            "TARGET SKILL:\n{}\n\nUSER RESPONSES:\n{}\n",
            skill.title, combined_input
        );

        let assessment: StoryAssessment = prompt_structured(
            model,
            STORY_ASSESS_PREAMBLE,
            &assessment_prompt,
            "story_assessment",
        ).await?;

        let computed_missing = missing_fields(&assessment.parsed_story);
        let mut missing_fields = assessment.missing_fields.clone();
        for field in computed_missing.iter() {
            if !missing_fields.contains(field) {
                missing_fields.push(field.clone());
            }
        }

        let mut action = assessment.action.clone();
        if matches!(action, NextAction::SaveStory) && !missing_fields.is_empty() {
            action = NextAction::AskFollowup;
        }

        if matches!(action, NextAction::SaveStory) {
            return Ok(Some(assessment.parsed_story));
        }

        if turn + 1 >= MAX_SKILL_TURNS {
            return Err(anyhow!(
                "Missing required details after follow-ups for skill: {}",
                skill.title
            ));
        }

        let question = match action {
            NextAction::AskAdjacent => {
                if assessment.adjacent_question.trim().is_empty() {
                    adjacent_fallback_question(skill)
                } else {
                    assessment.adjacent_question.clone()
                }
            }
            NextAction::AskFollowup => {
                if assessment.followup_question.trim().is_empty() {
                    followup_fallback_question(&missing_fields)
                } else {
                    assessment.followup_question.clone()
                }
            }
            NextAction::SaveStory => followup_fallback_question(&missing_fields),
        };

        println!("\n{}\n", question);
        io::stdout().flush().ok();
        let answer = read_multiline()?;
        if answer.trim().is_empty() {
            return Ok(None);
        }
        let label = match action {
            NextAction::AskAdjacent => "Adjacent experience answer",
            _ => "Follow-up answer",
        };
        combined_input = format!("{combined_input}\n\n{label}:\n{answer}");
    }

    Err(anyhow!(
        "Missing required details after follow-ups for skill: {}",
        skill.title
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    // 1) Load job posting text
    let job_path = args
        .get(1)
        .ok_or_else(|| anyhow!("usage: cargo run -- <job_text_file> [starter_resume_file]"))?;

    let job_raw = fs::read_to_string(job_path)
        .with_context(|| format!("failed to read file: {}", job_path))?;
    let job_text = normalize_whitespace(&job_raw);

    let starter_resume = args
        .get(2)
        .map(|path| {
            fs::read_to_string(path)
                .with_context(|| format!("failed to read file: {}", path))
        })
        .transpose()?;

    // 2) Load knowledge base + retrieve context
    let openai_client = openai_client_from_env()?;
    let embed_model = openai_client.embedding_model(EMBEDDING_MODEL_NAME);
    let completion_model = openai_client.completion_model(MODEL_NAME);

    // 3) Identify skill gaps with structured outputs
    let job_prompt = format!("JOB POSTING:\n{}\n", job_text);

    eprintln!("Requesting job needs from {MODEL_NAME}...");
    let job_needs: JobNeeds = prompt_structured(
        &completion_model,
        JOB_POST_PREAMBLE,
        &job_prompt,
        "job_needs_list",
    ).await?;

    let job_needs_json =
        serde_json::to_string_pretty(&job_needs).context("failed to serialize job needs")?;
    println!("\n=== Job Needs ===");
    for (idx, skill) in job_needs.skills.iter().enumerate() {
        // println!("{}", skill);
        println!("{}. {}{:?}", idx + 1, skill.need, skill.title);
        println!("\tdescription: {}", skill.description);
    }
    let matches = build_master::get_skills(job_text.clone()).await?;
    let context = matches
        .iter()
        .map(|(_score, _id, doc)| format!("- {doc}"))
        .collect::<Vec<_>>()
        .join("\n");

    // 3) Identify skill gaps with structured outputs
    let skill_plan_prompt = format!(
        "JOB NEEDS:\n{}\n\nRETRIEVED EVIDENCE:\n{}\n",
        job_needs_json, context
    );

    let skill_plan: SkillFocusList = prompt_structured(
        &completion_model,
        SKILL_PLAN_PREAMBLE,
        &skill_plan_prompt,
        "skill_focus_list",
    )
    .await?;

    if skill_plan.skills.is_empty() {
        println!("No skill gaps detected from the posting and existing evidence.");
        return Ok(());
    }

    println!("\n=== Skill Focus Areas ===");
    println!("Summary: {}\n\n", skill_plan.summary);
    for (idx, skill) in skill_plan.skills.iter().enumerate() {
        // println!("{}", skill);
        println!("{}. {:?}", idx + 1, skill.title);
        println!("\tdescription: {}", skill.description);
        println!("\tskill_description: {}", skill.skill_description);
        println!("\tsuitability: {}", skill.suitability);
        println!("\tneed: {}", skill.need);
        println!("\tjustification: {}", skill.justification);
    }

    // 4) Iterate through skills, collect stories, and save
    for (idx, skill) in skill_plan.skills.iter().enumerate() {
        println!(
            "\n=== Skill {}/{}: {} ===",
            idx + 1,
            skill_plan.skills.len(),
            skill.title
        );
    
        let parsed = collect_story_for_skill(&completion_model, skill).await?;
        let Some(parsed) = parsed else {
            println!("Skipped");
            continue;
        };
        let story = StorySeed {
            company: parsed.company,
            year: parsed.year,
            text: parsed.text,
        };

        build_master::add_story_to_store(story, &embed_model).await?;
        println!("Saved");
    }

    let kb_docs = build_master::list_story_documents()?;
    let kb_context = if kb_docs.is_empty() {
        "None.".to_string()
    } else {
        kb_docs
            .iter()
            .map(|doc| format!("- {doc}"))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let starter_resume = starter_resume
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "None provided.".to_string());

    let user_profile = match build_master::get_user_profile()? {
        Some(profile) => profile,
        None => {
            println!("\nNo user profile found in knowledge base. Let's add it.");
            let profile = collect_user_profile()?;
            build_master::set_user_profile(profile.clone())?;
            profile
        }
    };
    let user_profile_context = format_user_profile(&user_profile);

    let resume_prompt = format!(
        "JOB DESCRIPTION:\n{}\n\nKNOWLEDGE BASE:\n{}\n\nUSER PROFILE:\n{}\n\nSTARTER RESUME:\n{}\n\nTEMPLATE:\n{}\n",
        job_raw, kb_context, user_profile_context, starter_resume, RESUME_TEMPLATE_LATEX
    );

    let mut resume_latex =
        prompt_text(&completion_model, RESUME_BUILD_PREAMBLE, &resume_prompt).await?;
    let output_tex = "generated_resume.tex";

    for attempt in 0..=MAX_LATEX_FIXES {
        fs::write(output_tex, &resume_latex)
            .with_context(|| format!("failed to write resume to {}", output_tex))?;
        println!("\nWrote LaTeX resume to {}", output_tex);

        let compile = compile_latex(output_tex)?;
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
        resume_latex = prompt_text(&completion_model, RESUME_FIX_PREAMBLE, &fix_prompt).await?;
    }

    Ok(())
}
