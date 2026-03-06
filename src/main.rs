use std::fs;
use std::path::PathBuf;
use anyhow::Context;
use rig::completion::CompletionModel;
use rig::client::{CompletionClient, EmbeddingsClient};
use pipelines::llm::{self, Provider, NullEmbeddingModel};
use pipelines::{hiring_manager, kb, resume_builder, resume_coach};
use clap::Parser;

/// AI-powered resume builder. Given a job posting, analyzes required skills,
/// coaches you on gaps, and generates a tailored resume PDF.
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

    /// The directory to store the output files, including resume LaTeX and PDF
    #[arg(short, long, default_value = "./")]
    out: PathBuf,

    /// Also generate a cover letter
    #[arg(long, default_value_t = false)]
    cover_letter: bool,

    /// Minimum skill gap (need - suitability) to trigger coaching. Skills with
    /// a gap below this threshold are skipped during the coaching phase.
    #[arg(long, default_value_t = 3)]
    gap_threshold: i16,

    /// The path to the job posting as a text file
    job_path: PathBuf,
}

fn output_filename(job_text: &str) -> String {
    let words: Vec<&str> = job_text.split_whitespace().take(6).collect();
    let slug: String = words
        .join("_")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
        .collect::<String>()
        .to_lowercase();

    if slug.is_empty() {
        "generated_resume".to_string()
    } else {
        let truncated: String = slug.chars().take(60).collect();
        format!("resume_{}", truncated)
    }
}

async fn run_pipeline<M, E>(
    args: &Args,
    job_text: &String,
    model_name: &str,
    completion_model: &M,
    embed_model: &E,
) -> anyhow::Result<()>
where
    M: CompletionModel + Clone,
    E: rig::embeddings::EmbeddingModel,
{
    eprintln!("Requesting job needs from {model_name}...");
    let skill_assessment = hiring_manager::evaluate_candidate(job_text, completion_model, args.provider).await?;
    eprintln!("Done.");

    if skill_assessment.skills.is_empty() {
        println!("No skill gaps detected from the posting and existing evidence.");
    } else {
        println!("\n=== Skill Focus Areas ===");
        println!("Summary: {}\n\n", skill_assessment.summary);
        for (idx, skill) in skill_assessment.skills.iter().enumerate() {
            println!("{}. {:?}", idx + 1, skill.title);
            println!("\tdescription: {}", skill.description);
            println!("\tskill_description: {}", skill.skill_description);
            println!("\tsuitability: {}", skill.suitability);
            println!("\tneed: {}", skill.need);
            println!("\tjustification: {}", skill.justification);
        }

        resume_coach::fill_skill_gaps(&skill_assessment, args.gap_threshold, completion_model, embed_model, args.provider).await?;
    }

    let user_profile = kb::get_or_build_user_profile()?;
    let tex_filename = output_filename(job_text);

    resume_builder::build_resume(
        job_text,
        user_profile.clone(),
        &skill_assessment,
        &args.out,
        &tex_filename,
        completion_model,
        embed_model,
    ).await?;

    if args.cover_letter {
        resume_builder::build_cover_letter(
            job_text,
            &user_profile,
            &skill_assessment,
            &args.out,
            &tex_filename,
            completion_model,
            embed_model,
        ).await?;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let job_raw = fs::read_to_string(args.job_path.clone())
        .with_context(|| format!("failed to read file: {}", args.job_path.to_str().unwrap()))?;
    let job_text = &job_raw.split_whitespace().collect::<Vec<_>>().join(" ");

    let model_name = args.model.clone()
        .unwrap_or_else(|| args.provider.default_model().to_string());

    match args.provider {
        Provider::Claude => {
            let client = llm::anthropic_client_from_env()?;
            let completion_model = client.completion_model(&model_name);
            let embed = NullEmbeddingModel;
            run_pipeline(&args, job_text, &model_name, &completion_model, &embed).await
        }
        Provider::OpenAI => {
            let client = llm::openai_client_from_env()?;
            let completion_model = client.completion_model(&model_name);
            let embed_name = args.embedding_model.clone()
                .or_else(|| Provider::OpenAI.default_embedding_model().map(String::from))
                .unwrap();
            let embed_model = client.embedding_model(&embed_name);
            run_pipeline(&args, job_text, &model_name, &completion_model, &embed_model).await
        }
        Provider::Gemini => {
            let client = llm::gemini_client_from_env()?;
            let completion_model = client.completion_model(&model_name);
            let embed_name = args.embedding_model.clone()
                .or_else(|| Provider::Gemini.default_embedding_model().map(String::from))
                .unwrap();
            let embed_model = client.embedding_model(&embed_name);
            run_pipeline(&args, job_text, &model_name, &completion_model, &embed_model).await
        }
        Provider::Ollama => {
            let client = llm::ollama_client_from_env()?;
            let completion_model = client.completion_model(&model_name);
            if let Some(embed_name) = args.embedding_model.clone()
                .or_else(|| Provider::Ollama.default_embedding_model().map(String::from))
            {
                let embed_model = client.embedding_model(&embed_name);
                run_pipeline(&args, job_text, &model_name, &completion_model, &embed_model).await
            } else {
                let embed = NullEmbeddingModel;
                run_pipeline(&args, job_text, &model_name, &completion_model, &embed).await
            }
        }
        Provider::DeepSeek => {
            let client = llm::deepseek_client_from_env()?;
            let completion_model = client.completion_model(&model_name);
            let embed = NullEmbeddingModel;
            run_pipeline(&args, job_text, &model_name, &completion_model, &embed).await
        }
        Provider::Groq => {
            let client = llm::groq_client_from_env()?;
            let completion_model = client.completion_model(&model_name);
            let embed = NullEmbeddingModel;
            run_pipeline(&args, job_text, &model_name, &completion_model, &embed).await
        }
        Provider::XAI => {
            let client = llm::xai_client_from_env()?;
            let completion_model = client.completion_model(&model_name);
            let embed = NullEmbeddingModel;
            run_pipeline(&args, job_text, &model_name, &completion_model, &embed).await
        }
    }
}
