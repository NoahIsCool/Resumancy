use std::{fs, path::Path};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rig::client::{CompletionClient, EmbeddingsClient};
use rig::completion::CompletionModel;
use rig::embeddings::EmbeddingModel;
use pipelines::kb::{story_document, Story, UserSkillStore, UserSkillStoreSeed, DEFAULT_KB_PATH};
use pipelines::llm::{self, prompt_structured, CacheConfig, Provider, NullEmbeddingModel};

const KB_PREAMBLE: &str = r#"Role: You are a resume parsing assistant.
Goal: Build an initial skills knowledge base from the resume.
Rules:
- Quote the resume exactly; do not invent.
- Each skill entry must include:
  - company: inferred from the section (e.g., "research" for research experience); use "unknown" if unclear.
  - year: a 4-digit year; if missing, use the start year of the position; if unavailable, use "unknown".
  - text: the exact resume excerpt describing the skill or accomplishment.
"#;

/// Bootstrap the knowledge base from a resume text file.
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

    /// Path to the resume text file
    resume_path: String,

    /// Output path for the knowledge base JSON
    #[arg(default_value = DEFAULT_KB_PATH)]
    output_path: String,
}

async fn build_kb<M, E>(
    resume_raw: &str,
    model_name: &str,
    completion_model: &M,
    embed_model: &E,
    output_path: &str,
    provider: Provider,
    cache: Option<&CacheConfig>,
) -> Result<()>
where
    M: CompletionModel + Clone,
    E: EmbeddingModel,
{
    let prompt = format!("RESUME:\n{}\n", resume_raw);

    eprintln!("Parsing resume with {model_name}...");
    let result =
        prompt_structured(completion_model, KB_PREAMBLE, &prompt, "user_skill_store_seed", provider, cache).await?;
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
    let store = UserSkillStore {
        skills,
        user_profile: None,
    };

    let out_path = Path::new(output_path);
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create output dir: {}", parent.display()))?;
        }
    }
    let json = serde_json::to_string_pretty(&store)?;
    fs::write(out_path, json).with_context(|| {
        format!("failed to write knowledge base to {}", out_path.display())
    })?;

    println!(
        "Saved {} skills to {}",
        store.skills.len(),
        out_path.display()
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let resume_raw = fs::read_to_string(&args.resume_path)
        .with_context(|| format!("failed to read file: {}", args.resume_path))?;
    if resume_raw.trim().is_empty() {
        return Err(anyhow!("resume file is empty: {}", args.resume_path));
    }

    let model_name = args.model.clone()
        .unwrap_or_else(|| args.provider.default_model().to_string());

    let cache = CacheConfig {
        provider_name: args.provider.name().to_string(),
        model_name: model_name.clone(),
        enabled: !args.no_cache,
    };
    let cache_opt = Some(&cache);

    match args.provider {
        Provider::Claude => {
            let client = llm::anthropic_client_from_env()?;
            let model = client.completion_model(&model_name);
            let embed = NullEmbeddingModel;
            build_kb(&resume_raw, &model_name, &model, &embed, &args.output_path, args.provider, cache_opt).await
        }
        Provider::OpenAI => {
            let client = llm::openai_client_from_env()?;
            let model = client.completion_model(&model_name);
            let embed_name = args.embedding_model.clone()
                .or_else(|| Provider::OpenAI.default_embedding_model().map(String::from))
                .unwrap();
            let embed_model = client.embedding_model(&embed_name);
            build_kb(&resume_raw, &model_name, &model, &embed_model, &args.output_path, args.provider, cache_opt).await
        }
        Provider::Gemini => {
            let client = llm::gemini_client_from_env()?;
            let model = client.completion_model(&model_name);
            let embed_name = args.embedding_model.clone()
                .or_else(|| Provider::Gemini.default_embedding_model().map(String::from))
                .unwrap();
            let embed_model = client.embedding_model(&embed_name);
            build_kb(&resume_raw, &model_name, &model, &embed_model, &args.output_path, args.provider, cache_opt).await
        }
        Provider::Ollama => {
            let client = llm::ollama_client_from_env()?;
            let model = client.completion_model(&model_name);
            if let Some(embed_name) = args.embedding_model.clone()
                .or_else(|| Provider::Ollama.default_embedding_model().map(String::from))
            {
                let embed_model = client.embedding_model(&embed_name);
                build_kb(&resume_raw, &model_name, &model, &embed_model, &args.output_path, args.provider, cache_opt).await
            } else {
                let embed = NullEmbeddingModel;
                build_kb(&resume_raw, &model_name, &model, &embed, &args.output_path, args.provider, cache_opt).await
            }
        }
        Provider::DeepSeek => {
            let client = llm::deepseek_client_from_env()?;
            let model = client.completion_model(&model_name);
            let embed = NullEmbeddingModel;
            build_kb(&resume_raw, &model_name, &model, &embed, &args.output_path, args.provider, cache_opt).await
        }
        Provider::Groq => {
            let client = llm::groq_client_from_env()?;
            let model = client.completion_model(&model_name);
            let embed = NullEmbeddingModel;
            build_kb(&resume_raw, &model_name, &model, &embed, &args.output_path, args.provider, cache_opt).await
        }
        Provider::XAI => {
            let client = llm::xai_client_from_env()?;
            let model = client.completion_model(&model_name);
            let embed = NullEmbeddingModel;
            build_kb(&resume_raw, &model_name, &model, &embed, &args.output_path, args.provider, cache_opt).await
        }
    }
}
