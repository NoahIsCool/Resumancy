mod build_master;

use std::{
    env,
    fs,
    io::{self, BufRead, Write},
};
use anyhow::{anyhow, Context, Result};
use pipelines::kb::StorySeed;
use pipelines::llm::{prompt_structured, EMBEDDING_MODEL_NAME, MODEL_NAME};
use rig::client::{CompletionClient, EmbeddingsClient, ProviderClient};
use rig::completion::CompletionModel;
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const MAX_SKILL_TURNS: usize = 4;
const SKILL_PLAN_PREAMBLE: &str = r#"Role: You are a hiring manager. Given a job posting and retrieved evidence from a resume knowledge base, identify skills required by the posting, and how well the candidate satisfies each skill requirement. Identify every key word and industry skill mentioned. Be sure to "Read between the lines" to include skills that are implied, but not explicitly mentioned. For example, A role as an "AI Researger" would likely faveor a graduate degree in AI, and even though it is not explicitly mentioned, it would likely prefer experience pubishing research papers. Be sure to be very thurough and candid.

For each skill, return a struct with the following fields:
- title of the skill
- ranking from 0-9 of how nessasery the skill is to the job posting. 0 is not needed at all, 9 is a skill absolutely required, and anything less than 5 is simply nice to have.
- a ranking from 0-9 of how well the candidate satisfies the skill requirement. 0 is not listed and 9 satisfies the skill requirement fully.
- short description of the skill area.
- a short justifcation of the candidate's ranking in this skill area.


"#;

const STORY_ASSESS_PREAMBLE: &str = r#"You are a resume coach.
Given a target skill and the user's responses, decide the next action.
Rules:
- Always extract the story fields: company, year, text. Use empty strings for missing fields.
- Year should be a 4-digit year or "unknown" if unavailable.
- If the user states they have no direct experience, set action to "ask_adjacent" and provide one concise adjacent question.
- If any required fields are missing, set action to "ask_followup" and ask one concise follow-up question.
- If all required fields are present, set action to "save_story".
- missing_fields should list any missing required fields.
- If a question is not needed, set its field to an empty string."#;

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
struct SkillFocusList {
    skills: Vec<SkillNeed>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
struct SkillNeed {
    title: String, // title of the skill
    description: String,
    need: u8, //ranking from 0-9 of how necessary the skill is
    suitability: u8, // a ranking from 0-9 of how well the candidate satisfies requirement
    skill_description: String, // short description of the skill area.
    justification: String, // a short justification of the candidate's ranking in this skill area.
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[serde(rename_all = "snake_case")]
enum NextAction {
    SaveStory,
    AskFollowup,
    AskAdjacent,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
struct ParsedStory {
    company: String,
    year: String,
    text: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
struct StoryAssessment {
    action: NextAction,
    missing_fields: Vec<String>,
    followup_question: String,
    adjacent_question: String,
    parsed_story: ParsedStory,
}

fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn missing_fields(parsed: &ParsedStory) -> Vec<String> {
    let mut missing = Vec::new();
    if parsed.company.trim().is_empty() {
        missing.push("company".to_string());
    }
    if parsed.year.trim().is_empty() {
        missing.push("year".to_string());
    }
    if parsed.text.trim().is_empty() {
        missing.push("text".to_string());
    }
    missing
}

fn read_multiline() -> Result<String> {
    let mut input = String::new();
    let mut line = String::new();
    let stdin = io::stdin();
    let mut handle = stdin.lock();

    loop {
        line.clear();
        let bytes = handle.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
        if trimmed.is_empty() {
            break;
        }
        input.push_str(trimmed);
        input.push('\n');
    }

    Ok(input.trim().to_string())
}

fn followup_fallback_question(missing_fields: &[String]) -> String {
    let mut parts = Vec::new();
    if !missing_fields.is_empty() {
        parts.push(format!("missing fields: {}", missing_fields.join(", ")));
    }
    if parts.is_empty() {
        "Could you add a bit more detail to your story?".to_string()
    } else {
        format!("Could you add details for {}?", parts.join(" | "))
    }
}

fn adjacent_fallback_question(skill: &SkillNeed) -> String {
    format!(
        "If you don't have direct experience with {}, what adjacent experience could you share?",
        skill.title
    )
}

async fn collect_story_for_skill<M>(model: &M, skill: &SkillNeed) -> Result<ParsedStory>
where
    M: CompletionModel + Clone,
{
    println!("Target skill: {}", skill.title);
    println!("Skill description: {}", skill.skill_description);
    println!(
        "Share one story that demonstrates this skill. Include:\n\
- company\n- year (or \"unknown\")\n- story text (1-3 sentences)\n\
Finish with an empty line."
    );
    io::stdout().flush().ok();

    let mut combined_input = read_multiline()?;

    for turn in 0..MAX_SKILL_TURNS {
        let assessment_prompt = format!(
            "TARGET SKILL:\n{}\n\nUSER RESPONSES:\n{}\n",
            skill.title, combined_input
        );

        let assessment: StoryAssessment = prompt_structured(
            model,
            STORY_ASSESS_PREAMBLE,
            &assessment_prompt,
            "story_assessment",
        )
        .await?;

        let computed_missing = missing_fields(&assessment.parsed_story);
        let mut missing_fields = assessment.missing_fields.clone();
        for field in computed_missing.iter() {
            if !missing_fields.contains(field) {
                missing_fields.push(field.clone());
            }
        }

        let mut action = assessment.action.clone();
        if matches!(action, NextAction::SaveStory) && !missing_fields.is_empty() {
            action = NextAction::AskFollowup;
        }

        if matches!(action, NextAction::SaveStory) {
            return Ok(assessment.parsed_story);
        }

        if turn + 1 >= MAX_SKILL_TURNS {
            return Err(anyhow!(
                "Missing required details after follow-ups for skill: {}",
                skill.title
            ));
        }

        let question = match action {
            NextAction::AskAdjacent => {
                if assessment.adjacent_question.trim().is_empty() {
                    adjacent_fallback_question(skill)
                } else {
                    assessment.adjacent_question.clone()
                }
            }
            NextAction::AskFollowup => {
                if assessment.followup_question.trim().is_empty() {
                    followup_fallback_question(&missing_fields)
                } else {
                    assessment.followup_question.clone()
                }
            }
            NextAction::SaveStory => followup_fallback_question(&missing_fields),
        };

        println!("\n{}\n", question);
        io::stdout().flush().ok();
        let answer = read_multiline()?;
        let label = match action {
            NextAction::AskAdjacent => "Adjacent experience answer",
            _ => "Follow-up answer",
        };
        combined_input = format!("{combined_input}\n\n{label}:\n{answer}");
    }

    Err(anyhow!(
        "Missing required details after follow-ups for skill: {}",
        skill.title
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1) Load job posting text
    let job_path = env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: cargo run -- <job_text_file>"))?;

    let job_raw = fs::read_to_string(&job_path)
        .with_context(|| format!("failed to read file: {}", job_path))?;
    let job_text = normalize_whitespace(&job_raw);

    // 2) Load knowledge base + retrieve context
    let openai_client = openai::Client::from_env();
    let embed_model = openai_client.embedding_model(EMBEDDING_MODEL_NAME);
    let completion_model = openai_client.completion_model(MODEL_NAME);

    let matches = build_master::get_skills(job_text.clone()).await?;
    let context = matches
        .iter()
        .map(|(_score, _id, doc)| format!("- {doc}"))
        .collect::<Vec<_>>()
        .join("\n");

    // 3) Identify skill gaps with structured outputs
    let skill_plan_prompt = format!(
        "JOB POSTING:\n{}\n\nRETRIEVED EVIDENCE:\n{}\n",
        job_text, context
    );

    let skill_plan: SkillFocusList = prompt_structured(
        &completion_model,
        SKILL_PLAN_PREAMBLE,
        &skill_plan_prompt,
        "skill_focus_list",
    )
    .await?;

    if skill_plan.skills.is_empty() {
        println!("No skill gaps detected from the posting and existing evidence.");
        return Ok(());
    }

    println!("\n=== Skill Focus Areas ===");
    for (idx, skill) in skill_plan.skills.iter().enumerate() {
        // println!("{}", skill);
        println!("{}. {:?}", idx + 1, skill.title);
        println!("\tdescription: {}", skill.description);
        println!("\tskill_description: {}", skill.skill_description);
        println!("\tsuitability: {}", skill.suitability);
        println!("\tneed: {}", skill.need);
        println!("\tjustification: {}", skill.justification);
    }

    // 4) Iterate through skills, collect stories, and save
    for (idx, skill) in skill_plan.skills.iter().enumerate() {
        println!(
            "\n=== Skill {}/{}: {} ===",
            idx + 1,
            skill_plan.skills.len(),
            skill.title
        );
    
        let parsed = collect_story_for_skill(&completion_model, skill).await?;
        let story = StorySeed {
            company: parsed.company,
            year: parsed.year,
            text: parsed.text,
        };

        build_master::add_story_to_store(story, &embed_model).await?;
        println!("Saved");
    }

    Ok(())
}
