use std::fs;
use std::path::PathBuf;
use anyhow::Context;
use pipelines::llm::{
    openai_client_from_env, EMBEDDING_MODEL_NAME, MODEL_NAME,
};
use rig::client::{CompletionClient, EmbeddingsClient};
use pipelines::{hiring_manager, kb, resume_builder, resume_coach};
use clap::Parser;

/// This is a resume building applicaiont
#[derive(Parser, Debug)]
#[command(version, about, long_about)]
struct Args {
    // the directory to store the output files, including resume latex and pdf
    #[arg(short, long, default_value = "./")]
    out: PathBuf,

    /// The path to the job posting as a text file
    job_path: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // 1) Load job posting text
    let job_raw = fs::read_to_string(args.job_path.clone())
        .with_context(|| format!("failed to read file: {}", args.job_path.to_str().unwrap()))?;
    let job_text = &job_raw.split_whitespace().collect::<Vec<_>>().join(" ");


    // 2) Load knowledge base + retrieve context
    let openai_client = openai_client_from_env()?;
    let embed_model = openai_client.embedding_model(EMBEDDING_MODEL_NAME);
    let completion_model = openai_client.completion_model(MODEL_NAME);

    eprintln!("Requesting job needs from {MODEL_NAME}...");
    let skill_assessment = hiring_manager::evaluate_candidate(job_text, &completion_model).await?;
    eprintln!("Done.");

    if skill_assessment.skills.is_empty() {
        println!("No skill gaps detected from the posting and existing evidence.");
        return Ok(());
    }
    println!("\n=== Skill Focus Areas ===");
    println!("Summary: {}\n\n", skill_assessment.summary);
    for (idx, skill) in skill_assessment.skills.iter().enumerate() {
        // println!("{}", skill);
        println!("{}. {:?}", idx + 1, skill.title);
        println!("\tdescription: {}", skill.description);
        println!("\tskill_description: {}", skill.skill_description);
        println!("\tsuitability: {}", skill.suitability);
        println!("\tneed: {}", skill.need);
        println!("\tjustification: {}", skill.justification);
    }

    resume_coach::fill_skill_gaps(skill_assessment, &completion_model, &embed_model).await?;

    let user_profile = kb::get_or_build_user_profile()?;

    resume_builder::build_resume(job_text, user_profile, &args.out, &completion_model).await?;

    Ok(())
}
