use std::fs;
use std::path::{Path, PathBuf};
use anyhow::{anyhow, Context};
use rig::completion::CompletionModel;
use rig::client::{CompletionClient, EmbeddingsClient};
use pipelines::llm::{self, Provider, NullEmbeddingModel, CacheConfig};
use pipelines::stats::{StatsCollector, LlmCall};
use pipelines::ui::Ui;
use pipelines::{hiring_manager, kb, resume_builder, resume_coach};
use clap::Parser;

/// AI-powered resume builder. Given a job posting (or a folder of postings),
/// analyzes required skills, coaches you on gaps, and generates a tailored
/// resume PDF.
///
/// Single file: pipelines job.txt
/// Batch mode:  pipelines jobs/   (recursively finds all .md files)
#[derive(Parser, Debug)]
#[command(version, about, long_about)]
struct Args {
    /// LLM provider to use
    #[arg(long, value_enum, env = "LLM_PROVIDER", default_value_t = Provider::Claude)]
    provider: Provider,

    /// Model name override (defaults to provider's recommended model)
    #[arg(long, env = "LLM_MODEL")]
    model: Option<String>,

    /// Embedding model name override
    #[arg(long, env = "LLM_EMBEDDING_MODEL")]
    embedding_model: Option<String>,

    /// The directory to store output files (single-file mode only; batch mode
    /// writes to an out/ folder next to each job file)
    #[arg(short, long, default_value = "./")]
    out: PathBuf,

    /// Also generate a cover letter
    #[arg(long, default_value_t = false)]
    cover_letter: bool,

    /// Run self-evaluation loop (LLM-as-judge) on generated resume
    #[arg(long, default_value_t = false)]
    eval: bool,

    /// Print LLM usage statistics after completion
    #[arg(long, default_value_t = false)]
    stats: bool,

    /// Minimum skill gap (need - suitability) to trigger coaching. Skills with
    /// a gap below this threshold are skipped during the coaching phase.
    #[arg(long, default_value_t = 3)]
    gap_threshold: i16,

    /// Disable response caching
    #[arg(long, default_value_t = false)]
    no_cache: bool,

    /// Stream LLM output for resume generation
    #[arg(long, default_value_t = false)]
    stream: bool,

    /// Enable OpenTelemetry tracing (sends to OTLP endpoint)
    #[arg(long, default_value_t = false)]
    trace: bool,

    /// Path to a job posting file, or a directory to batch-process (finds all
    /// .md files recursively)
    job_path: PathBuf,
}

fn init_tracing(trace: bool) -> Option<opentelemetry_sdk::trace::SdkTracerProvider> {
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::EnvFilter;

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("warn"));

    if trace {
        use opentelemetry::trace::TracerProvider;

        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .build()
            .expect("failed to create OTLP exporter");

        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .build();

        let tracer = provider.tracer("pipelines");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer().with_target(false).with_writer(std::io::stderr))
            .with(otel_layer)
            .init();

        Some(provider)
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer().with_target(false).with_writer(std::io::stderr))
            .init();

        None
    }
}

/// Recursively collect all `.md` files under `dir`.
fn find_md_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut results = Vec::new();
    find_md_files_recursive(dir, &mut results)?;
    results.sort();
    Ok(results)
}

fn find_md_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir)
        .with_context(|| format!("failed to read directory: {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            find_md_files_recursive(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "md") {
            out.push(path);
        }
    }
    Ok(())
}

/// Run the full pipeline for a single job posting.
async fn run_pipeline<M, E>(
    job_text: &String,
    out_dir: &Path,
    model_name: &str,
    completion_model: &M,
    embed_model: &E,
    provider: Provider,
    cover_letter: bool,
    run_eval: bool,
    stream: bool,
    gap_threshold: i16,
    ui: &Ui,
    stats: &StatsCollector,
    cache: &CacheConfig,
) -> anyhow::Result<()>
where
    M: CompletionModel + Clone,
    E: rig::embeddings::EmbeddingModel,
{
    let cache_opt = Some(cache);

    ui.header("Analyzing job posting");

    let spinner = ui.spinner(&format!("Requesting job needs from {}...", model_name));
    let timer = stats.start_timer();
    let skill_assessment = hiring_manager::evaluate_candidate(job_text, completion_model, provider, cache_opt).await?;
    stats.record(LlmCall {
        label: "evaluate_candidate".into(),
        model: model_name.into(),
        prompt_tokens: 0,
        completion_tokens: 0,
        duration: timer.elapsed(),
    });
    spinner.finish("Job analysis complete");

    if skill_assessment.skills.is_empty() {
        ui.success("No skill gaps detected from the posting and existing evidence.");
    } else {
        ui.header("Skill Focus Areas");
        ui.detail("Summary", &skill_assessment.summary);
        ui.divider();
        for (idx, skill) in skill_assessment.skills.iter().enumerate() {
            println!(
                "  {}. {} (need: {}, suitability: {})",
                idx + 1, skill.title, skill.need, skill.suitability
            );
            println!("     {}", skill.justification);
        }

        ui.header("Coaching on skill gaps");
        let timer = stats.start_timer();
        resume_coach::fill_skill_gaps(&skill_assessment, gap_threshold, completion_model, embed_model, provider).await?;
        stats.record(LlmCall {
            label: "skill_coaching".into(),
            model: model_name.into(),
            prompt_tokens: 0,
            completion_tokens: 0,
            duration: timer.elapsed(),
        });
    }

    let user_profile = kb::get_or_build_user_profile()?;

    // Resume generation — use streaming or spinner
    let spinner = if stream {
        ui.header("Generating resume");
        None
    } else {
        Some(ui.spinner("Building tailored resume..."))
    };

    let timer = stats.start_timer();
    let evaluation = resume_builder::build_resume(
        job_text,
        user_profile.clone(),
        &skill_assessment,
        out_dir,
        "resume",
        completion_model,
        embed_model,
        provider,
        run_eval,
        stream,
        cache_opt,
    ).await?;
    stats.record(LlmCall {
        label: "build_resume".into(),
        model: model_name.into(),
        prompt_tokens: 0,
        completion_tokens: 0,
        duration: timer.elapsed(),
    });

    if let Some(s) = spinner {
        s.finish("Resume generated");
    } else {
        ui.success("Resume generated");
    }

    if let Some(eval) = &evaluation {
        ui.header("Resume Evaluation");
        ui.detail("Score", &format!("{}/9", eval.overall_score));
        if !eval.strengths.is_empty() {
            println!("  Strengths:");
            for s in &eval.strengths {
                println!("    + {}", s);
            }
        }
        if !eval.weaknesses.is_empty() {
            println!("  Weaknesses:");
            for w in &eval.weaknesses {
                println!("    - {}", w);
            }
        }
    }

    if cover_letter {
        let cl_spinner = if stream {
            ui.header("Writing cover letter");
            None
        } else {
            Some(ui.spinner("Writing cover letter..."))
        };

        let timer = stats.start_timer();
        resume_builder::build_cover_letter(
            job_text,
            &user_profile,
            &skill_assessment,
            out_dir,
            "cover",
            completion_model,
            embed_model,
            stream,
            cache_opt,
        ).await?;
        stats.record(LlmCall {
            label: "cover_letter".into(),
            model: model_name.into(),
            prompt_tokens: 0,
            completion_tokens: 0,
            duration: timer.elapsed(),
        });

        if let Some(s) = cl_spinner {
            s.finish("Cover letter generated");
        } else {
            ui.success("Cover letter generated");
        }
    }

    Ok(())
}

/// Dispatch to the correct provider and run the pipeline for a single job.
async fn dispatch_pipeline(
    args: &Args,
    job_text: &String,
    out_dir: &Path,
    model_name: &str,
    gap_threshold: i16,
    ui: &Ui,
    stats: &StatsCollector,
    cache: &CacheConfig,
) -> anyhow::Result<()> {
    match args.provider {
        Provider::Claude => {
            let client = llm::anthropic_client_from_env()?;
            let m = client.completion_model(model_name);
            let e = NullEmbeddingModel;
            run_pipeline(job_text, out_dir, model_name, &m, &e, args.provider, args.cover_letter, args.eval, args.stream, gap_threshold, ui, stats, cache).await
        }
        Provider::OpenAI => {
            let client = llm::openai_client_from_env()?;
            let m = client.completion_model(model_name);
            let embed_name = args.embedding_model.clone()
                .or_else(|| Provider::OpenAI.default_embedding_model().map(String::from))
                .unwrap();
            let e = client.embedding_model(&embed_name);
            run_pipeline(job_text, out_dir, model_name, &m, &e, args.provider, args.cover_letter, args.eval, args.stream, gap_threshold, ui, stats, cache).await
        }
        Provider::Gemini => {
            let client = llm::gemini_client_from_env()?;
            let m = client.completion_model(model_name);
            let embed_name = args.embedding_model.clone()
                .or_else(|| Provider::Gemini.default_embedding_model().map(String::from))
                .unwrap();
            let e = client.embedding_model(&embed_name);
            run_pipeline(job_text, out_dir, model_name, &m, &e, args.provider, args.cover_letter, args.eval, args.stream, gap_threshold, ui, stats, cache).await
        }
        Provider::Ollama => {
            let client = llm::ollama_client_from_env()?;
            let m = client.completion_model(model_name);
            if let Some(embed_name) = args.embedding_model.clone()
                .or_else(|| Provider::Ollama.default_embedding_model().map(String::from))
            {
                let e = client.embedding_model(&embed_name);
                run_pipeline(job_text, out_dir, model_name, &m, &e, args.provider, args.cover_letter, args.eval, args.stream, gap_threshold, ui, stats, cache).await
            } else {
                let e = NullEmbeddingModel;
                run_pipeline(job_text, out_dir, model_name, &m, &e, args.provider, args.cover_letter, args.eval, args.stream, gap_threshold, ui, stats, cache).await
            }
        }
        Provider::DeepSeek => {
            let client = llm::deepseek_client_from_env()?;
            let m = client.completion_model(model_name);
            let e = NullEmbeddingModel;
            run_pipeline(job_text, out_dir, model_name, &m, &e, args.provider, args.cover_letter, args.eval, args.stream, gap_threshold, ui, stats, cache).await
        }
        Provider::Groq => {
            let client = llm::groq_client_from_env()?;
            let m = client.completion_model(model_name);
            let e = NullEmbeddingModel;
            run_pipeline(job_text, out_dir, model_name, &m, &e, args.provider, args.cover_letter, args.eval, args.stream, gap_threshold, ui, stats, cache).await
        }
        Provider::XAI => {
            let client = llm::xai_client_from_env()?;
            let m = client.completion_model(model_name);
            let e = NullEmbeddingModel;
            run_pipeline(job_text, out_dir, model_name, &m, &e, args.provider, args.cover_letter, args.eval, args.stream, gap_threshold, ui, stats, cache).await
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let tracer_provider = init_tracing(args.trace);

    let ui = Ui::new(console::Term::stderr().is_term());
    let stats = StatsCollector::new(args.stats);

    let model_name = args.model.clone()
        .unwrap_or_else(|| args.provider.default_model().to_string());

    let cache = CacheConfig {
        provider_name: args.provider.name().to_string(),
        model_name: model_name.clone(),
        enabled: !args.no_cache,
    };

    ui.detail("Provider", &format!("{:?}", args.provider));
    ui.detail("Model", &model_name);
    if cache.enabled {
        ui.detail("Cache", "enabled");
    }

    let result = if args.job_path.is_dir() {
        // ── Batch mode ──────────────────────────────────────────────
        let md_files = find_md_files(&args.job_path)?;
        if md_files.is_empty() {
            return Err(anyhow!("no .md files found in {}", args.job_path.display()));
        }

        ui.header(&format!("Batch mode: {} job(s) found", md_files.len()));
        for path in &md_files {
            ui.detail("  Job", &path.display().to_string());
        }

        let mut errors: Vec<(PathBuf, anyhow::Error)> = Vec::new();

        for (idx, job_file) in md_files.iter().enumerate() {
            let job_dir = job_file.parent().unwrap_or(Path::new("."));
            let stem = job_file.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "out".to_string());
            let out_dir = job_dir.join(&stem);

            ui.divider();
            ui.header(&format!(
                "[{}/{}] {}",
                idx + 1,
                md_files.len(),
                job_file.display()
            ));

            let job_raw = match fs::read_to_string(job_file) {
                Ok(raw) => raw,
                Err(e) => {
                    ui.error(&format!("failed to read {}: {}", job_file.display(), e));
                    errors.push((job_file.clone(), e.into()));
                    continue;
                }
            };
            let job_text = job_raw.split_whitespace().collect::<Vec<_>>().join(" ");

            match dispatch_pipeline(&args, &job_text, &out_dir, &model_name, args.gap_threshold, &ui, &stats, &cache).await {
                Ok(()) => ui.success(&format!("Done  -> {}", out_dir.display())),
                Err(e) => {
                    ui.error(&format!("Failed: {}", e));
                    errors.push((job_file.clone(), e));
                }
            }
        }

        if errors.is_empty() {
            ui.divider();
            ui.success(&format!("All {} job(s) completed successfully", md_files.len()));
            Ok(())
        } else {
            ui.divider();
            ui.error(&format!("{}/{} job(s) failed:", errors.len(), md_files.len()));
            for (path, err) in &errors {
                ui.detail("  Failed", &format!("{}: {}", path.display(), err));
            }
            Err(anyhow!("{} job(s) failed during batch processing", errors.len()))
        }
    } else {
        // ── Single-file mode ────────────────────────────────────────
        let job_raw = fs::read_to_string(&args.job_path)
            .with_context(|| format!("failed to read file: {}", args.job_path.display()))?;
        let job_text = job_raw.split_whitespace().collect::<Vec<_>>().join(" ");

        dispatch_pipeline(&args, &job_text, &args.out, &model_name, args.gap_threshold, &ui, &stats, &cache).await
    };

    if stats.enabled() {
        ui.divider();
        ui.stats_block(&stats.summary());
    }

    if let Some(provider) = tracer_provider {
        let _ = provider.shutdown();
    }

    result
}
