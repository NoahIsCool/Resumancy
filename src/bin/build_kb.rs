use std::{env, fs, path::Path};

use anyhow::{anyhow, Context, Result};
use pipelines::kb::{story_document, Story, UserSkillStore, UserSkillStoreSeed, DEFAULT_KB_PATH};
use pipelines::llm::{openai_client_from_env, prompt_structured, EMBEDDING_MODEL_NAME, MODEL_NAME};
use rig::client::{CompletionClient, EmbeddingsClient};
use rig::embeddings::EmbeddingModel;

const KB_PREAMBLE: &str = r#"Role: You are a resume parsing assistant.
Goal: Build an initial skills knowledge base from the resume.
Rules:
- Quote the resume exactly; do not invent.
- Each skill entry must include:
  - company: inferred from the section (e.g., "research" for research experience); use "unknown" if unclear.
  - year: a 4-digit year; if missing, use the start year of the position; if unavailable, use "unknown".
  - text: the exact resume excerpt describing the skill or accomplishment.
"#;

#[tokio::main]
async fn main() -> Result<()> {
    let resume_path = env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: cargo run --bin build_kb -- <resume_text_file> [output_path]"))?;
    let output_path = env::args()
        .nth(2)
        .unwrap_or_else(|| DEFAULT_KB_PATH.to_string());

    let resume_raw = fs::read_to_string(&resume_path)
        .with_context(|| format!("failed to read file: {}", resume_path))?;
    if resume_raw.trim().is_empty() {
        return Err(anyhow!("resume file is empty: {}", resume_path));
    }

    let prompt = format!("RESUME:\n{}\n", resume_raw);
    let openai_client = openai_client_from_env()?;
    let completion_model = openai_client.completion_model(MODEL_NAME);
    let embed_model = openai_client.embedding_model(EMBEDDING_MODEL_NAME);

    eprintln!("Parsing resume with {MODEL_NAME}...");
    let seed_store: UserSkillStoreSeed =
        prompt_structured(&completion_model, KB_PREAMBLE, &prompt, "user_skill_store_seed").await?;
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

    let out_path = Path::new(&output_path);
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create output dir: {}", parent.display()))?;
        }
    }
    let json = serde_json::to_string_pretty(&store)?;
    fs::write(out_path, json).with_context(|| {
        format!(
            "failed to write knowledge base to {}",
            out_path.display()
        )
    })?;

    println!(
        "Saved {} skills to {}",
        store.skills.len(),
        out_path.display()
    );
    Ok(())
}
