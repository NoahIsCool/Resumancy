use std::{fs, path::Path};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rig::completion::CompletionModel;
use rig::embeddings::EmbeddingModel;
use pipelines::input::InputEditor;
use pipelines::kb::{
    story_document, EducationEntry, JobEntry, ProfileLink, Story, UserProfile, UserSkillStore,
    UserSkillStoreSeed,
};
use pipelines::llm::{prompt_structured, CacheConfig, Provider};

const KB_PREAMBLE: &str = r#"Role: You are a resume parsing assistant.
Goal: Build an initial skills knowledge base from the resume.
Rules:
- Quote the resume exactly; do not invent.
- Each skill entry must include:
  - company: inferred from the section (e.g., "research" for research experience); use "unknown" if unclear.
  - year: a 4-digit year; if missing, use the start year of the position; if unavailable, use "unknown".
  - text: the exact resume excerpt describing the skill or accomplishment.
"#;

const PROFILE_PREAMBLE: &str = r#"Role: You are a resume parsing assistant.
Goal: Extract the candidate's profile information from the resume.
Rules:
- Extract only information explicitly present in the resume.
- Use empty string "" for any field not found.
- For links, look for GitHub, LinkedIn, personal websites, etc.
- For jobs, extract company, title, location, start_date, end_date.
- For education, extract degree title and graduation date.
"#;

/// Bootstrap the knowledge base from a resume file (PDF, LaTeX, Markdown, or plain text).
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// LLM provider to use
    #[arg(long, value_enum, env = "LLM_PROVIDER", default_value_t = Provider::Claude)]
    provider: Provider,

    /// Model name override
    #[arg(long, env = "LLM_MODEL")]
    model: Option<String>,

    /// Embedding model name override
    #[arg(long, env = "LLM_EMBEDDING_MODEL")]
    embedding_model: Option<String>,

    /// Disable response caching
    #[arg(long, default_value_t = false)]
    no_cache: bool,

    /// Cache TTL in seconds (entries older than this are treated as misses)
    #[arg(long)]
    cache_ttl: Option<u64>,

    /// Path to the resume file (PDF, .tex, .md, or .txt)
    resume_path: String,

    /// Output path for the knowledge base JSON (defaults to XDG data dir)
    output_path: Option<String>,
}

fn read_resume(path: &str) -> Result<String> {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    match ext {
        "pdf" => extract_pdf_text(path),
        _ => fs::read_to_string(path).with_context(|| format!("failed to read resume file: {path}")),
    }
}

fn extract_pdf_text(path: &str) -> Result<String> {
    let mut doc = pdf_oxide::PdfDocument::open(path)
        .map_err(|e| anyhow!("failed to open PDF: {e}"))?;
    let page_count = doc
        .page_count()
        .map_err(|e| anyhow!("failed to get page count: {e}"))?;
    let mut text = String::new();
    for i in 0..page_count {
        let page_text = doc
            .extract_text(i)
            .map_err(|e| anyhow!("failed to extract text from page {}: {e}", i + 1))?;
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&page_text);
    }
    Ok(text)
}

fn display_profile(profile: &UserProfile) {
    println!("\nParsed profile:");
    println!("  Name: {}", profile.name);
    println!("  Email: {}", profile.email);
    println!("  Phone: {}", profile.phone);
    println!("  Location: {}", profile.location);

    if profile.links.is_empty() {
        println!("\n  Links: (none)");
    } else {
        println!("\n  Links:");
        for (i, link) in profile.links.iter().enumerate() {
            println!("    {}. {}: {}", i + 1, link.label, link.url);
        }
    }

    if profile.education.is_empty() {
        println!("\n  Education: (none)");
    } else {
        println!("\n  Education:");
        for (i, edu) in profile.education.iter().enumerate() {
            println!("    {}. {} ({})", i + 1, edu.degree, edu.graduation_date);
        }
    }

    if profile.jobs.is_empty() {
        println!("\n  Jobs: (none)");
    } else {
        println!("\n  Jobs:");
        for (i, job) in profile.jobs.iter().enumerate() {
            println!(
                "    {}. {} — {}, {} ({} – {})",
                i + 1,
                job.company,
                job.title,
                job.location,
                job.start_date,
                job.end_date
            );
        }
    }
}

fn edit_profile(profile: &UserProfile, editor: &mut InputEditor) -> Result<UserProfile> {
    println!("\nEditing profile (press Enter to keep current value):");

    let name = edit_field(editor, "Name", &profile.name)?;
    let email = edit_field(editor, "Email", &profile.email)?;
    let phone = edit_field(editor, "Phone", &profile.phone)?;
    let location = edit_field(editor, "Location", &profile.location)?;

    println!("\n--- Links ---");
    let mut links = profile.links.clone();
    for (i, link) in links.iter().enumerate() {
        println!("  {}. {}: {}", i + 1, link.label, link.url);
    }
    loop {
        let label = editor.read_line("Add link label (blank to finish): ")?;
        let Some(label) = label else { break };
        let url = editor.read_required_line(&format!("URL for {}: ", label))?;
        links.push(ProfileLink { label, url });
    }

    println!("\n--- Education ---");
    let mut education = profile.education.clone();
    for (i, edu) in education.iter().enumerate() {
        println!("  {}. {} ({})", i + 1, edu.degree, edu.graduation_date);
    }
    loop {
        let degree = editor.read_line("Add degree (blank to finish): ")?;
        let Some(degree) = degree else { break };
        let graduation_date = editor.read_required_line("Graduation date: ")?;
        education.push(EducationEntry {
            degree,
            graduation_date,
        });
    }

    println!("\n--- Jobs ---");
    let mut jobs = profile.jobs.clone();
    for (i, job) in jobs.iter().enumerate() {
        println!(
            "  {}. {} — {}, {} ({} – {})",
            i + 1, job.company, job.title, job.location, job.start_date, job.end_date
        );
    }
    loop {
        let company = editor.read_line("Add company (blank to finish): ")?;
        let Some(company) = company else { break };
        let title = editor.read_required_line("Job title: ")?;
        let job_location = editor.read_required_line("Job location: ")?;
        let start_date = editor.read_required_line("Start date: ")?;
        let end_date = editor.read_required_line("End date (or Present): ")?;
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

fn edit_field(editor: &mut InputEditor, label: &str, current: &str) -> Result<String> {
    let prompt = format!("{} [{}]: ", label, current);
    match editor.read_line(&prompt)? {
        Some(new_val) => Ok(new_val),
        None => Ok(current.to_string()),
    }
}

fn confirm_profile(profile: &UserProfile, editor: &mut InputEditor) -> Result<Option<UserProfile>> {
    display_profile(profile);
    println!();
    let response = editor.read_line("Accept this profile? [Y/n/edit]: ")?;
    match response.as_deref() {
        None | Some("y") | Some("Y") | Some("yes") => Ok(Some(profile.clone())),
        Some("n") | Some("N") | Some("no") => {
            println!("Skipping profile (skills only).");
            Ok(None)
        }
        Some("edit") | Some("e") => {
            let edited = edit_profile(profile, editor)?;
            Ok(Some(edited))
        }
        _ => {
            println!("Unrecognized input, accepting as-is.");
            Ok(Some(profile.clone()))
        }
    }
}

async fn build_kb<M, E>(
    resume_raw: &str,
    model_name: &str,
    completion_model: &M,
    embed_model: &E,
    output_path: &Path,
    provider: Provider,
    cache: Option<&CacheConfig>,
) -> Result<()>
where
    M: CompletionModel + Clone,
    E: EmbeddingModel,
{
    let prompt = format!("RESUME:\n{}\n", resume_raw);

    // 1. Extract skills
    eprintln!("Parsing skills with {model_name}...");
    let result = prompt_structured(
        completion_model,
        KB_PREAMBLE,
        &prompt,
        "user_skill_store_seed",
        provider,
        cache,
    )
    .await?;
    let seed_store: UserSkillStoreSeed = result.value;
    let mut skills = Vec::with_capacity(seed_store.skills.len());
    for seed in seed_store.skills {
        let document = story_document(&seed.company, &seed.year, &seed.text);
        let embedding = embed_model.embed_text(&document).await?;
        skills.push(Story {
            company: seed.company,
            year: seed.year,
            text: seed.text,
            vector: embedding.vec,
        });
    }
    eprintln!("Extracted {} skills.", skills.len());

    // 2. Extract profile
    eprintln!("Extracting profile with {model_name}...");
    let profile_result = prompt_structured::<_, UserProfile>(
        completion_model,
        PROFILE_PREAMBLE,
        &prompt,
        "user_profile",
        provider,
        cache,
    )
    .await?;
    let extracted_profile = profile_result.value;

    // 3. Interactive confirm/edit
    let mut editor = InputEditor::new()?;
    let user_profile = confirm_profile(&extracted_profile, &mut editor)?;

    // 4. Save
    let store = UserSkillStore {
        embedding_model: Some(model_name.to_string()),
        skills,
        user_profile,
    };

    if let Some(parent) = output_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output dir: {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&store)?;
    fs::write(output_path, json).with_context(|| {
        format!(
            "failed to write knowledge base to {}",
            output_path.display()
        )
    })?;

    println!(
        "Saved {} skills to {}",
        store.skills.len(),
        output_path.display()
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Ensure data dirs exist
    let data_dir = pipelines::paths::data_dir()?;
    pipelines::paths::ensure_dir(&data_dir)?;
    pipelines::paths::ensure_dir(&pipelines::paths::cache_dir()?)?;

    pipelines::llm::enable_spinners(console::Term::stderr().is_term());

    let resume_raw = read_resume(&args.resume_path)?;
    if resume_raw.trim().is_empty() {
        return Err(anyhow!("resume file is empty: {}", args.resume_path));
    }

    let output_path = match &args.output_path {
        Some(p) => std::path::PathBuf::from(p),
        None => pipelines::paths::kb_path()?,
    };

    let model_name = args
        .model
        .clone()
        .unwrap_or_else(|| args.provider.default_model().to_string());

    let cache = CacheConfig {
        provider_name: args.provider.name().to_string(),
        model_name: model_name.clone(),
        enabled: !args.no_cache,
        max_age: args.cache_ttl.map(std::time::Duration::from_secs),
    };
    let cache_opt = Some(&cache);

    pipelines::dispatch_provider!(args.provider, &model_name, args.embedding_model.clone(), |model, embed| {
        build_kb(&resume_raw, &model_name, &model, &embed, &output_path, args.provider, cache_opt).await
    })
}
