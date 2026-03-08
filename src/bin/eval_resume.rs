use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rig::completion::CompletionModel;
use pipelines::eval::{evaluate_resume, ResumeEvaluation};
use pipelines::llm::{CacheConfig, Provider};

/// Evaluate a resume against a job posting using an LLM-as-judge.
///
/// Accepts a resume file (LaTeX, PDF, or plain text) and a job posting file,
/// then scores how well the resume targets the job.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// LLM provider to use
    #[arg(long, value_enum, env = "LLM_PROVIDER", default_value_t = Provider::Claude)]
    provider: Provider,

    /// Model name override
    #[arg(long, env = "LLM_MODEL")]
    model: Option<String>,

    /// Disable response caching
    #[arg(long, default_value_t = false)]
    no_cache: bool,

    /// Cache TTL in seconds (entries older than this are treated as misses)
    #[arg(long)]
    cache_ttl: Option<u64>,

    /// Output results as JSON instead of human-readable text
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Path to the resume file (.tex, .pdf, .md, or .txt)
    resume_path: String,

    /// Path to the job posting file
    job_path: String,
}

fn read_file(path: &str) -> Result<String> {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    match ext {
        "pdf" => extract_pdf_text(path),
        _ => fs::read_to_string(path).with_context(|| format!("failed to read file: {path}")),
    }
}

fn extract_pdf_text(path: &str) -> Result<String> {
    let mut doc =
        pdf_oxide::PdfDocument::open(path).map_err(|e| anyhow!("failed to open PDF: {e}"))?;
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

fn print_evaluation(eval: &ResumeEvaluation) {
    println!("Score: {}/9", eval.overall_score);

    if !eval.strengths.is_empty() {
        println!("\nStrengths:");
        for s in &eval.strengths {
            println!("  + {s}");
        }
    }

    if !eval.weaknesses.is_empty() {
        println!("\nWeaknesses:");
        for w in &eval.weaknesses {
            println!("  - {w}");
        }
    }

    if !eval.suggestions.is_empty() {
        println!("\nSuggestions:");
        for s in &eval.suggestions {
            println!("  * {s}");
        }
    }
}

async fn run_eval<M: CompletionModel + Clone>(
    resume_text: &str,
    job_text: &str,
    model: &M,
    provider: Provider,
    cache: Option<&CacheConfig>,
    json_output: bool,
) -> Result<()> {
    let eval = evaluate_resume(resume_text, job_text, model, provider, cache).await?;

    if json_output {
        println!("{}", serde_json::to_string_pretty(&eval)?);
    } else {
        print_evaluation(&eval);
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    pipelines::llm::enable_spinners(console::Term::stderr().is_term());

    let resume_text = read_file(&args.resume_path)?;
    if resume_text.trim().is_empty() {
        return Err(anyhow!("resume file is empty: {}", args.resume_path));
    }

    let job_text = read_file(&args.job_path)?;
    if job_text.trim().is_empty() {
        return Err(anyhow!("job file is empty: {}", args.job_path));
    }

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

    eprintln!("Evaluating resume with {} ({})...", args.provider.name(), model_name);

    pipelines::dispatch_completion!(args.provider, &model_name, |m| {
        run_eval(&resume_text, &job_text, &m, args.provider, cache_opt, args.json).await
    })
}
