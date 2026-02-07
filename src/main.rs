mod build_master;

use std::{env, fs, io::{self, Read}};
use anyhow::{anyhow, Context, Result};
use rig::{
    completion::Prompt,
    providers::openai,
};
use rig::client::{CompletionClient, EmbeddingsClient, ProviderClient};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Deserialize)]
struct ParsedSkill {
    title: Option<String>,
    company: Option<String>,
    position: Option<String>,
    section: Option<String>,
    story: Option<String>,
}

fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn missing_fields(parsed: &ParsedSkill) -> Vec<&'static str> {
    let mut missing = Vec::new();
    if parsed.title.as_deref().unwrap_or("").is_empty() { missing.push("title"); }
    if parsed.company.as_deref().unwrap_or("").is_empty() { missing.push("company"); }
    if parsed.position.as_deref().unwrap_or("").is_empty() { missing.push("position"); }
    if parsed.section.as_deref().unwrap_or("").is_empty() { missing.push("section"); }
    if parsed.story.as_deref().unwrap_or("").is_empty() { missing.push("story"); }
    missing
}

fn read_multiline() -> Result<String> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    Ok(input.trim().to_string())
}


fn skill_id_from_fields(parsed: &ParsedSkill) -> String {
    let mut hasher = Sha256::new();
    let joined = format!(
        "{}|{}|{}|{}|{}",
        parsed.title.as_deref().unwrap_or(""),
        parsed.company.as_deref().unwrap_or(""),
        parsed.position.as_deref().unwrap_or(""),
        parsed.section.as_deref().unwrap_or(""),
        parsed.story.as_deref().unwrap_or(""),
    );
    hasher.update(joined.as_bytes());
    format!("skill_{}", hex::encode(hasher.finalize()))
}

#[tokio::main]
async fn main() -> Result<()> {
    // Global subscriber must only be initialized once
    // tracing_subscriber::fmt()
    //     .with_env_filter(tracing_subscriber::EnvFilter::new("rig=trace"))
    //     .init();

    // 1) Load job posting text
    let job_path = env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: cargo run -- <job_text_file>"))?;

    let job_raw = fs::read_to_string(&job_path)
        .with_context(|| format!("failed to read file: {}", job_path))?;
    let job_text = normalize_whitespace(&job_raw);

    // 2) Load knowledge base + retrieve context
    let openai_client = openai::Client::from_env();
    let embed_model = openai_client.embedding_model("text-embedding-ada-002");

    let matches = build_master::get_skills(job_text.clone()).await?;
    let context = matches
        .iter()
        .map(|(_score, _id, doc)| format!("- {doc}"))
        .collect::<Vec<_>>()
        .join("\n");

    // 3) Ask model to identify weak points
    let analysis_agent = openai_client
        .agent("gpt-4o")
        .preamble(
"Role: You are a candid career coach.
Given a job posting and retrieved evidence from a knowledge base, list ALL key weak points where evidence is missing or weak.
Be concise, specific, and list only weak points.",
        )
        .build();

    let analysis_prompt = format!(
        "JOB POSTING:\n{}\n\nRETRIEVED EVIDENCE:\n{}\n",
        job_text, context
    );

    let weak_points = analysis_agent.prompt(&analysis_prompt).await?;
    println!("\n=== Key Weak Points ===\n{}\n", weak_points);

    // 4) Ask user for a story with guidance
    println!(
"Please share a story that demonstrates one of those weak points.

Be specific and include:
- title (what the skill/experience is called)
- company
- position
- section (e.g., \"Work Experience\", \"Projects\", \"Research\", \"Leadership\")
- story (what you did, how, impact/metrics, tools, constraints)

Tips: include measurable impact, your role, and the hardest technical detail.

Paste your story below. When finished, press Ctrl+D (macOS/Linux) or Ctrl+Z then Enter (Windows)."
    );

    let user_story = read_multiline()?;

    // 5) Parse fields from user story (with follow-ups if needed)
    let parse_agent = openai_client
        .agent("gpt-4o")
        .preamble(
"Extract the fields from the user's story. Respond ONLY with JSON:
{
  \"title\": \"...\",
  \"company\": \"...\",
  \"position\": \"...\",
  \"section\": \"...\",
  \"story\": \"...\"
}
If a field is missing, set it to an empty string.",
        )
        .build();

    let mut combined_input = user_story.clone();
    let mut parsed: ParsedSkill = serde_json::from_str(
        &parse_agent.prompt(&combined_input).await?
    )?;

    for _ in 0..3 {
        let missing = missing_fields(&parsed);
        if missing.is_empty() {
            break;
        }

        let followup_agent = openai_client
            .agent("gpt-4o")
            .preamble(
"Ask ONE concise follow-up question to collect the missing fields listed.",
            )
            .build();

        let followup_prompt = format!(
            "Missing fields: {:?}\nAsk a single follow-up question.",
            missing
        );

        let question = followup_agent.prompt(&followup_prompt).await?;
        println!("\n{}\n", question);
        let answer = read_multiline()?;
        combined_input = format!("{combined_input}\n\nFollow-up answer:\n{answer}");

        parsed = serde_json::from_str(
            &parse_agent.prompt(&combined_input).await?
        )?;
    }

    let missing = missing_fields(&parsed);
    if !missing.is_empty() {
        return Err(anyhow!("Missing required fields after follow-ups: {:?}", missing));
    }

    // 6) Save to knowledge base
    let skill = build_master::SkillDoc {
        id: skill_id_from_fields(&parsed),
        title: parsed.title.unwrap(),
        company: parsed.company.unwrap(),
        position: parsed.position.unwrap(),
        section: parsed.section.unwrap(),
        story: parsed.story.unwrap(),
    };

    build_master::add_story_to_store(skill, &embed_model).await?;
    println!("Saved");

    Ok(())
}
