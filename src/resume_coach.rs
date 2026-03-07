use std::io;
use std::io::{BufRead, Write};
use anyhow::anyhow;
use rig::completion::CompletionModel;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use crate::{kb, prompts};
use crate::hiring_manager::{SkillFocusList, SkillNeed};
use crate::kb::StorySeed;
use crate::llm::{prompt_structured, Provider};

const MAX_SKILL_TURNS: usize = 4;

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[serde(rename_all = "snake_case")]
enum NextAction {
    SaveStory,
    AskFollowup,
    AskAdjacent,
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

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
struct ParsedStory {
    company: String,
    year: String,
    text: String,
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


fn read_multiline() -> anyhow::Result<String> {
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    read_multiline_from(&mut handle)
}

fn read_multiline_from<R: BufRead>(handle: &mut R) -> anyhow::Result<String> {
    let mut input = String::new();
    let mut line = String::new();

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

async fn collect_story_for_skill<M>(
    model: &M,
    skill: &SkillNeed,
    related_skills: usize,
    embed_model: &impl rig::embeddings::EmbeddingModel,
    provider: Provider,
) -> anyhow::Result<Option<ParsedStory>>
where
    M: CompletionModel + Clone,
{
    println!("Target skill: {}", skill.title);
    println!("Skill description: {}", skill.skill_description);

    if related_skills > 0 {
        let query = format!("{}: {}", skill.title, skill.skill_description);
        let stories = kb::retrieve_relevant_stories(&query, related_skills, embed_model).await?;
        if !stories.is_empty() {
            println!("\nClosest existing stories in your knowledge base:");
            for (i, story) in stories.iter().enumerate() {
                println!("  {}. {} ({}): {}", i + 1, story.company, story.year, story.text);
            }
        }
    }

    println!(
        "\nShare one story that demonstrates this skill. Include:\n\
- company\n- year (or \"unknown\")\n- story text (1-3 sentences)\n\
Finish with an empty line."
    );
    io::stdout().flush().ok();

    let mut combined_input = read_multiline()?;
    if combined_input.trim().is_empty() {
        return Ok(None);
    }

    for turn in 0..MAX_SKILL_TURNS {
        let assessment_prompt = format!(
            "TARGET SKILL:\n{}\n\nUSER RESPONSES:\n{}\n",
            skill.title, combined_input
        );

        let assessment: StoryAssessment = prompt_structured(
            model,
            prompts::STORY_ASSESS_PREAMBLE,
            &assessment_prompt,
            "story_assessment",
            provider,
            None,
        ).await?.value;

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
            return Ok(Some(assessment.parsed_story));
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
        if answer.trim().is_empty() {
            return Ok(None);
        }
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

#[tracing::instrument(skip_all)]
pub async fn fill_skill_gaps<M: CompletionModel + Clone>(
    skill_assessment: &SkillFocusList,
    gap_threshold: i16,
    related_skills: usize,
    completion_model: &M,
    embed_model: &impl rig::embeddings::EmbeddingModel,
    provider: Provider,
) -> Result<(), anyhow::Error> {
    let gaps: Vec<_> = skill_assessment.skills.iter()
        .filter(|s| (s.need as i16 - s.suitability as i16) >= gap_threshold)
        .collect();

    if gaps.is_empty() {
        println!("No significant skill gaps to address (threshold: {}).", gap_threshold);
        return Ok(());
    }

    println!("\nCoaching on {} skill gap(s) (need - suitability >= {}):", gaps.len(), gap_threshold);

    for (idx, skill) in gaps.iter().enumerate() {
        println!(
            "\n=== Skill {}/{}: {} (need: {}, suitability: {}) ===",
            idx + 1,
            gaps.len(),
            skill.title,
            skill.need,
            skill.suitability,
        );

        let parsed = collect_story_for_skill(completion_model, skill, related_skills, embed_model, provider).await?;
        let Some(parsed) = parsed else {
            println!("Skipped");
            continue;
        };
        let story = StorySeed {
            company: parsed.company,
            year: parsed.year,
            text: parsed.text,
        };

        kb::add_story_to_store(story, embed_model).await?;
        println!("Saved");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hiring_manager::SkillNeed;
    use std::io::Cursor;

    #[test]
    fn missing_fields_detects_empty_values() {
        let parsed = ParsedStory {
            company: "".to_string(),
            year: "  ".to_string(),
            text: "\n".to_string(),
        };
        let fields = missing_fields(&parsed);
        assert_eq!(fields, vec!["company", "year", "text"]);
    }

    #[test]
    fn missing_fields_returns_empty_when_complete() {
        let parsed = ParsedStory {
            company: "Acme".to_string(),
            year: "2022".to_string(),
            text: "Did stuff".to_string(),
        };
        let fields = missing_fields(&parsed);
        assert!(fields.is_empty());
    }

    #[test]
    fn followup_fallback_question_without_missing() {
        let question = followup_fallback_question(&[]);
        assert_eq!(question, "Could you add a bit more detail to your story?");
    }

    #[test]
    fn followup_fallback_question_with_missing() {
        let missing = vec!["company".to_string(), "year".to_string()];
        let question = followup_fallback_question(&missing);
        assert_eq!(
            question,
            "Could you add details for missing fields: company, year?"
        );
    }

    #[test]
    fn adjacent_fallback_question_uses_title() {
        let skill = SkillNeed {
            title: "Rust".to_string(),
            description: "Systems programming".to_string(),
            need: 8,
            suitability: 4,
            skill_description: "Rust language".to_string(),
            justification: "Some experience".to_string(),
        };
        let question = adjacent_fallback_question(&skill);
        assert!(question.contains("Rust"));
    }

    #[test]
    fn read_multiline_from_stops_on_blank_line() {
        let input = "line1\nline2\n\nline3\n";
        let mut reader = Cursor::new(input);
        let output = read_multiline_from(&mut reader).expect("read");
        assert_eq!(output, "line1\nline2");
    }

    #[test]
    fn read_multiline_from_reads_until_eof() {
        let input = "line1\r\nline2";
        let mut reader = Cursor::new(input);
        let output = read_multiline_from(&mut reader).expect("read");
        assert_eq!(output, "line1\nline2");
    }
}
