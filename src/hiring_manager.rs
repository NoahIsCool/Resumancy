use anyhow::Context;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use rig::completion::CompletionModel;
use crate::llm::{prompt_structured, Provider};
use crate::prompts;
use crate::kb::load_kb;

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
pub struct JobNeeds {
    pub skills: Vec<Need>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
pub struct Need {
    pub title: String, // title of the skill
    pub description: String, // short description of the skill area.
    pub skill_description: String, // short phrase describing the skill area.
    pub need: u8, //ranking from 0-9 of how necessary the skill is
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
pub struct SkillFocusList {
    pub summary: String,
    pub skills: Vec<SkillNeed>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
pub struct SkillNeed {
    pub title: String, // title of the skill
    pub description: String,
    pub need: u8, //ranking from 0-9 of how necessary the skill is
    pub suitability: u8, // a ranking from 0-9 of how well the candidate satisfies requirement
    pub skill_description: String, // short description of the skill area.
    pub justification: String, // a short justification of the candidate's ranking in this skill area.
}

pub async fn get_job_needs<M: CompletionModel + Clone>(job_text: &String, completion_model: &M, provider: Provider) -> Result<JobNeeds, anyhow::Error> {
    let job_prompt = format!("JOB POSTING:\n{}\n", job_text);

    let job_needs: JobNeeds = prompt_structured(
        completion_model,
        prompts::JOB_POST_PREAMBLE,
        &job_prompt,
        "job_needs_list",
        provider,
    ).await?;

    Ok(job_needs)
}

pub async fn evaluate_candidate<M: CompletionModel + Clone>(job_text: &String, completion_model: &M, provider: Provider) -> Result<SkillFocusList, anyhow::Error> {
    let job_needs = get_job_needs(job_text, completion_model, provider).await?;

    let job_needs_str =
        serde_json::to_string_pretty(&job_needs).context("failed to serialize job needs")?;
    for (idx, skill) in job_needs.skills.iter().enumerate() {
        // println!("{}", skill);
        println!("{}. {}{:?}", idx + 1, skill.need, skill.title);
        println!("\tdescription: {}", skill.description);
    }


    let kb = load_kb()?.skills;
    let context = kb
        .iter()
        .map(|skill | format!("- {} ({}): {}", skill.company, skill.year, skill.text))
        .collect::<Vec<_>>()
        .join("\n");

    // 3) Identify skill gaps with structured outputs
    let skill_plan_prompt = format!(
        "JOB NEEDS:\n{}\n\nRETRIEVED EVIDENCE:\n{}\n",
        job_needs_str, context
    );

    let skill_plan: SkillFocusList = prompt_structured(
        completion_model,
        prompts::EVALUATION_PREAMBLE,
        &skill_plan_prompt,
        "skill_focus_list",
        provider,
    ).await?;
    
    Ok(skill_plan)
}
