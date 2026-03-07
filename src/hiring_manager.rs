use anyhow::Context;
use rig::completion::{CompletionModel, Usage};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use crate::llm::{prompt_structured, combine_usage, CacheConfig, Provider};
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

pub async fn get_job_needs<M: CompletionModel + Clone>(
    job_text: &String,
    completion_model: &M,
    provider: Provider,
    cache: Option<&CacheConfig>,
) -> Result<(JobNeeds, Usage), anyhow::Error> {
    let job_prompt = format!("JOB POSTING:\n{}\n", job_text);

    let result = prompt_structured(
        completion_model,
        prompts::JOB_POST_PREAMBLE,
        &job_prompt,
        "job_needs_list",
        provider,
        cache,
    ).await?;

    Ok((result.value, result.usage))
}

#[tracing::instrument(skip_all)]
pub async fn evaluate_candidate<M: CompletionModel + Clone>(
    job_text: &String,
    completion_model: &M,
    provider: Provider,
    cache: Option<&CacheConfig>,
) -> Result<(SkillFocusList, Usage), anyhow::Error> {
    let (job_needs, usage1) = get_job_needs(job_text, completion_model, provider, cache).await?;

    let job_needs_str =
        serde_json::to_string_pretty(&job_needs).context("failed to serialize job needs")?;
    for (idx, skill) in job_needs.skills.iter().enumerate() {
        println!("{}. {}{:?}", idx + 1, skill.need, skill.title);
        println!("\tdescription: {}", skill.description);
    }


    let kb = load_kb()?;
    let stories_context = kb.skills
        .iter()
        .map(|skill| format!("- {} ({}): {}", skill.company, skill.year, skill.text))
        .collect::<Vec<_>>()
        .join("\n");

    let profile_context = match &kb.user_profile {
        Some(p) => {
            let mut parts = Vec::new();
            for edu in &p.education {
                parts.push(format!("- Education: {} ({})", edu.degree, edu.graduation_date));
            }
            for job in &p.jobs {
                parts.push(format!("- {} — {}, {} ({} to {})", job.company, job.title, job.location, job.start_date, job.end_date));
            }
            parts.join("\n")
        }
        None => String::new(),
    };

    let mut context = stories_context;
    if !profile_context.is_empty() {
        context = format!("{}\n\n{}", profile_context, context);
    }

    // 3) Identify skill gaps with structured outputs
    let skill_plan_prompt = format!(
        "JOB NEEDS:\n{}\n\nRETRIEVED EVIDENCE:\n{}\n",
        job_needs_str, context
    );

    let result = prompt_structured(
        completion_model,
        prompts::EVALUATION_PREAMBLE,
        &skill_plan_prompt,
        "skill_focus_list",
        provider,
        cache,
    ).await?;

    Ok((result.value, combine_usage(usage1, result.usage)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_needs_deserializes() {
        let json = r#"{
            "skills": [
                {
                    "title": "Rust",
                    "description": "Systems programming",
                    "skill_description": "Rust language proficiency",
                    "need": 8
                },
                {
                    "title": "Python",
                    "description": "Scripting",
                    "skill_description": "Python scripting",
                    "need": 5
                }
            ]
        }"#;
        let needs: JobNeeds = serde_json::from_str(json).expect("parse");
        assert_eq!(needs.skills.len(), 2);
        assert_eq!(needs.skills[0].title, "Rust");
        assert_eq!(needs.skills[0].need, 8);
        assert_eq!(needs.skills[1].title, "Python");
    }

    #[test]
    fn skill_focus_list_deserializes() {
        let json = r#"{
            "summary": "Strong candidate overall",
            "skills": [
                {
                    "title": "ML",
                    "description": "Machine learning",
                    "need": 9,
                    "suitability": 7,
                    "skill_description": "ML engineering",
                    "justification": "Good research background"
                }
            ]
        }"#;
        let list: SkillFocusList = serde_json::from_str(json).expect("parse");
        assert_eq!(list.summary, "Strong candidate overall");
        assert_eq!(list.skills.len(), 1);
        assert_eq!(list.skills[0].need, 9);
        assert_eq!(list.skills[0].suitability, 7);
    }

    #[test]
    fn skill_need_fields_correctly_typed() {
        let json = r#"{
            "title": "Go",
            "description": "Backend development",
            "need": 6,
            "suitability": 3,
            "skill_description": "Go programming",
            "justification": "Limited experience"
        }"#;
        let skill: SkillNeed = serde_json::from_str(json).expect("parse");
        assert_eq!(skill.title, "Go");
        assert_eq!(skill.need, 6);
        assert_eq!(skill.suitability, 3);
        assert_eq!(skill.justification, "Limited experience");
    }
}
