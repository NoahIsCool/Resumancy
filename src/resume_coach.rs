use anyhow::anyhow;
use rig::completion::CompletionModel;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use crate::{kb, prompts};
use crate::hiring_manager::{SkillFocusList, SkillNeed};
use crate::input::InputEditor;
use crate::kb::StorySeed;
use crate::llm::{prompt_structured, Provider};

const MAX_SKILL_TURNS: usize = 4;
const DUPLICATE_SIMILARITY_THRESHOLD: f64 = 0.85;
const RELATED_DISPLAY_MIN: usize = 3;

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
    coach_message: String,
    missing_fields: Vec<String>,
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


fn fallback_question(skill: &SkillNeed, missing_fields: &[String], action: &NextAction) -> String {
    match action {
        NextAction::AskAdjacent => {
            format!(
                "If you don't have direct experience with {}, what adjacent experience could you share?",
                skill.title
            )
        }
        _ => {
            if !missing_fields.is_empty() {
                format!("Could you add details for {}?", missing_fields.join(", "))
            } else {
                "Could you add a bit more detail to your story?".to_string()
            }
        }
    }
}

fn print_commands() {
    println!("  (Enter twice on blank line to submit, or type a /command to submit immediately)");
    println!("  Commands: blank line = skip | /done = save as-is | /use N = use existing story | /edit N = edit existing story");
}

fn confirm_or_edit_story(
    parsed: &ParsedStory,
    combined_input: &str,
    editor: &mut InputEditor,
) -> anyhow::Result<Option<ParsedStory>> {
    println!("\nHere's what I captured:");
    println!("  Company: {}", parsed.company);
    println!("  Year: {}", parsed.year);
    println!("  Story: {}", parsed.text);

    let response = editor
        .read_line("\nSave this story? [Y/n/edit] ")?
        .unwrap_or_default()
        .to_lowercase();

    if response == "n" {
        return Ok(None);
    } else if response == "edit" {
        return edit_parsed_story(parsed, editor);
    }

    // Default: accept as-is. If parsed story is mostly empty, save the raw input instead.
    if parsed.text.trim().is_empty() && !combined_input.trim().is_empty() {
        return Ok(Some(ParsedStory {
            company: if parsed.company.trim().is_empty() {
                "unknown".to_string()
            } else {
                parsed.company.clone()
            },
            year: if parsed.year.trim().is_empty() {
                "unknown".to_string()
            } else {
                parsed.year.clone()
            },
            text: combined_input.trim().to_string(),
        }));
    }

    Ok(Some(parsed.clone()))
}

fn edit_parsed_story(
    parsed: &ParsedStory,
    editor: &mut InputEditor,
) -> anyhow::Result<Option<ParsedStory>> {
    println!("\nCurrent story text: {}", parsed.text);
    println!("Type your replacement (blank line to keep current):");
    let edited_text = editor.read_multiline("")?;
    let text = edited_text.unwrap_or_else(|| parsed.text.clone());

    let company = editor
        .read_line(&format!("Company [{}]: ", parsed.company))?
        .unwrap_or_else(|| parsed.company.clone());
    let year = editor
        .read_line(&format!("Year [{}]: ", parsed.year))?
        .unwrap_or_else(|| parsed.year.clone());

    Ok(Some(ParsedStory {
        company,
        year,
        text,
    }))
}

/// Display related stories from the KB. Returns them for later reference by index.
async fn show_related_stories(
    skill: &SkillNeed,
    related_skills: usize,
    embed_model: &impl rig::embeddings::EmbeddingModel,
) -> anyhow::Result<Vec<kb::Story>> {
    let display_count = related_skills.max(RELATED_DISPLAY_MIN);
    let query = format!("{}: {}", skill.title, skill.skill_description);
    let stories = kb::retrieve_relevant_stories(&query, display_count, embed_model).await?;
    if !stories.is_empty() {
        println!("\nRelated stories in your knowledge base:");
        for (i, story) in stories.iter().enumerate() {
            println!("  {}. {} ({}): {}", i + 1, story.company, story.year, story.text);
        }
    }
    Ok(stories)
}

/// Parse a `/use N` or `/edit N` command, returning the 0-based index.
fn parse_command_index(input: &str, prefix: &str, max: usize) -> Option<usize> {
    let rest = input.strip_prefix(prefix)?.trim();
    let n: usize = rest.parse().ok()?;
    if n >= 1 && n <= max { Some(n - 1) } else { None }
}

/// Handle `/edit N` — edit an existing KB story in-place.
async fn handle_edit_existing(
    index: usize,
    stories: &[kb::Story],
    embed_model: &impl rig::embeddings::EmbeddingModel,
    editor: &mut InputEditor,
) -> anyhow::Result<()> {
    let story = &stories[index];
    println!("\nEditing story {}:", index + 1);
    println!("  Company: {}", story.company);
    println!("  Year: {}", story.year);
    println!("  Story: {}", story.text);

    println!("\nType new story text (blank line to keep current):");
    let edited_text = editor.read_multiline("")?;
    let text = edited_text.unwrap_or_else(|| story.text.clone());

    let company = editor
        .read_line(&format!("Company [{}]: ", story.company))?
        .unwrap_or_else(|| story.company.clone());
    let year = editor
        .read_line(&format!("Year [{}]: ", story.year))?
        .unwrap_or_else(|| story.year.clone());

    // Find the actual KB index for this story (stories list may be a subset)
    let store = kb::load_kb()?;
    let kb_index = store.skills.iter().position(|s| {
        s.company == story.company && s.year == story.year && s.text == story.text
    });

    if let Some(kb_idx) = kb_index {
        kb::update_story(kb_idx, StorySeed { company, year, text }, embed_model).await?;
        println!("Updated story in knowledge base.");
    } else {
        println!("Could not find story in KB to update — saving as new instead.");
        kb::add_story_to_store(StorySeed { company, year, text }, embed_model).await?;
    }

    Ok(())
}

/// Check if the new story is a near-duplicate of an existing one.
/// If so, offer to replace, merge, or save as new.
/// Returns `true` if the story was handled (saved/replaced/skipped), `false` to proceed normally.
async fn check_and_handle_duplicate(
    parsed: &ParsedStory,
    embed_model: &impl rig::embeddings::EmbeddingModel,
    editor: &mut InputEditor,
) -> anyhow::Result<Option<bool>> {
    let document = kb::story_document(&parsed.company, &parsed.year, &parsed.text);
    let embedding = embed_model.embed_text(&document).await?;
    let store = kb::load_kb()?;
    let similar = kb::find_similar_stories(&store, &embedding.vec, DUPLICATE_SIMILARITY_THRESHOLD);

    if similar.is_empty() {
        return Ok(None); // no duplicate, proceed normally
    }

    println!("\nThis looks similar to existing stories:");
    for (i, (_, score, story)) in similar.iter().enumerate() {
        println!(
            "  {}. [{:.0}% match] {} ({}): {}",
            i + 1,
            score * 100.0,
            story.company,
            story.year,
            story.text
        );
    }

    println!("\nOptions:");
    println!("  [S]ave as new  — add alongside the existing story");
    println!("  [r]eplace N    — replace existing story N with your new version");
    println!("  [k]eep existing — skip saving (existing story already covers this)");
    println!("  [e]dit         — edit your new story before saving");

    let response = editor
        .read_line("\nChoice [S/r N/k/e]: ")?
        .unwrap_or_default();
    let trimmed = response.trim().to_lowercase();

    if trimmed == "k" || trimmed == "keep" {
        println!("Keeping existing story.");
        return Ok(Some(true)); // handled, skip save
    }

    if trimmed == "e" || trimmed == "edit" {
        return Ok(Some(false)); // not handled, caller should run edit flow
    }

    if trimmed.starts_with('r') {
        // Parse "r N" or "replace N"
        let num_part = trimmed.trim_start_matches("replace").trim_start_matches('r').trim();
        let n: usize = num_part.parse().unwrap_or(1);
        if n >= 1 && n <= similar.len() {
            let (kb_idx, _, _) = similar[n - 1];
            let seed = StorySeed {
                company: parsed.company.clone(),
                year: parsed.year.clone(),
                text: parsed.text.clone(),
            };
            kb::update_story(kb_idx, seed, embed_model).await?;
            println!("Replaced existing story.");
            return Ok(Some(true));
        } else {
            println!("Invalid number, saving as new.");
        }
    }

    Ok(None) // save as new (default)
}

enum UserInput {
    /// User provided story text (with any trailing command stripped).
    Text(String),
    Skip,
    /// /done — save what we have so far.
    Done,
    /// /done after text — save text as-is, skip LLM coaching.
    DoneWithText(String),
    UseExisting(usize),
    EditExisting(usize),
}

/// Check if the last non-empty line of `text` is a command.
/// If so, return `(remaining_text, command_line)`.
fn split_trailing_command(text: &str) -> Option<(String, String)> {
    let lines: Vec<&str> = text.lines().collect();
    // Find the last non-empty line
    let last = lines.iter().rposition(|l| !l.trim().is_empty())?;
    let last_line = lines[last].trim();
    if last_line.starts_with('/') {
        let before = lines[..last].join("\n");
        let before_trimmed = before.trim().to_string();
        Some((before_trimmed, last_line.to_string()))
    } else {
        None
    }
}

fn read_user_input(editor: &mut InputEditor, related_count: usize) -> anyhow::Result<UserInput> {
    let Some(text) = editor.read_multiline("")? else {
        return Ok(UserInput::Skip);
    };
    let trimmed = text.trim();

    // Pure command (no preceding text)
    if trimmed == "/done" {
        return Ok(UserInput::Done);
    }
    if let Some(idx) = parse_command_index(trimmed, "/use", related_count) {
        return Ok(UserInput::UseExisting(idx));
    }
    if let Some(idx) = parse_command_index(trimmed, "/edit", related_count) {
        return Ok(UserInput::EditExisting(idx));
    }

    // Check if a command is on the last line after story text
    if let Some((before, cmd)) = split_trailing_command(trimmed) {
        if cmd == "/done" {
            if before.is_empty() {
                return Ok(UserInput::Done);
            }
            return Ok(UserInput::DoneWithText(before));
        }
        if let Some(idx) = parse_command_index(&cmd, "/use", related_count) {
            return Ok(UserInput::UseExisting(idx));
        }
        if let Some(idx) = parse_command_index(&cmd, "/edit", related_count) {
            return Ok(UserInput::EditExisting(idx));
        }
    }

    Ok(UserInput::Text(text))
}

async fn collect_story_for_skill<M>(
    model: &M,
    skill: &SkillNeed,
    related_skills: usize,
    embed_model: &impl rig::embeddings::EmbeddingModel,
    provider: Provider,
    editor: &mut InputEditor,
    is_additional: bool,
) -> anyhow::Result<Option<ParsedStory>>
where
    M: CompletionModel + Clone,
{
    if !is_additional {
        println!("Target skill: {}", skill.title);
        println!("Skill description: {}", skill.skill_description);
    }

    let related = show_related_stories(skill, related_skills, embed_model).await?;

    if is_additional {
        println!("\nAdd another story for {}.", skill.title);
    } else {
        println!("\nTell me about a time you demonstrated this skill.");
    }
    print_commands();

    let initial = match read_user_input(editor, related.len())? {
        UserInput::Skip => return Ok(None),
        UserInput::Done => return Ok(None),
        UserInput::DoneWithText(text) => {
            // User typed text + /done — save directly, skip LLM coaching
            let story = ParsedStory {
                company: "unknown".to_string(),
                year: "unknown".to_string(),
                text,
            };
            return confirm_or_edit_story(&story, "", editor);
        }
        UserInput::UseExisting(idx) => {
            println!("Using existing story: {} ({}): {}", related[idx].company, related[idx].year, related[idx].text);
            return Ok(None);
        }
        UserInput::EditExisting(idx) => {
            handle_edit_existing(idx, &related, embed_model, editor).await?;
            return Ok(None);
        }
        UserInput::Text(text) => text,
    };
    let mut combined_input = initial;

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
        let mut missing = assessment.missing_fields.clone();
        for field in computed_missing.iter() {
            if !missing.contains(field) {
                missing.push(field.clone());
            }
        }

        let mut action = assessment.action.clone();
        if matches!(action, NextAction::SaveStory) && !missing.is_empty() {
            action = NextAction::AskFollowup;
        }

        if matches!(action, NextAction::SaveStory) {
            return confirm_or_edit_story(&assessment.parsed_story, &combined_input, editor);
        }

        if turn + 1 >= MAX_SKILL_TURNS {
            return confirm_or_edit_story(&assessment.parsed_story, &combined_input, editor);
        }

        if !assessment.coach_message.trim().is_empty() {
            println!("\n{}", assessment.coach_message);
        } else {
            println!("\n{}", fallback_question(skill, &missing, &action));
        }
        print_commands();

        match read_user_input(editor, related.len())? {
            UserInput::Skip => return Ok(None),
            UserInput::Done => {
                return confirm_or_edit_story(&assessment.parsed_story, &combined_input, editor);
            }
            UserInput::DoneWithText(extra) => {
                // User added more context + /done — include it before saving
                combined_input = format!("{combined_input}\n\nFollow-up answer:\n{extra}");
                return confirm_or_edit_story(&assessment.parsed_story, &combined_input, editor);
            }
            UserInput::UseExisting(idx) => {
                println!("Using existing story: {} ({}): {}", related[idx].company, related[idx].year, related[idx].text);
                return Ok(None);
            }
            UserInput::EditExisting(idx) => {
                handle_edit_existing(idx, &related, embed_model, editor).await?;
                return Ok(None);
            }
            UserInput::Text(answer) => {
                let label = match action {
                    NextAction::AskAdjacent => "Adjacent experience answer",
                    _ => "Follow-up answer",
                };
                combined_input = format!("{combined_input}\n\n{label}:\n{answer}");
            }
        }
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
    editor: &mut InputEditor,
) -> Result<(), anyhow::Error> {
    let gaps: Vec<_> = skill_assessment.skills.iter()
        .filter(|s| (s.need as i16 - s.suitability as i16) >= gap_threshold)
        .collect();

    if gaps.is_empty() {
        println!("No significant skill gaps to address (threshold: {}).", gap_threshold);
        return Ok(());
    }

    let kb_total = kb::load_kb()?.skills.len();
    println!("\nCoaching on {} skill gap(s) (need - suitability >= {}). Knowledge base has {} stories.", gaps.len(), gap_threshold, kb_total);

    for (idx, skill) in gaps.iter().enumerate() {
        println!(
            "\n=== Skill {}/{}: {} (need: {}, suitability: {}) ===",
            idx + 1,
            gaps.len(),
            skill.title,
            skill.need,
            skill.suitability,
        );

        let mut story_count = 0;
        loop {
            let parsed = collect_story_for_skill(
                completion_model, skill, related_skills, embed_model,
                provider, editor, story_count > 0,
            ).await?;
            let Some(parsed) = parsed else {
                if story_count == 0 {
                    println!("Skipped");
                }
                break;
            };
            let story = StorySeed {
                company: parsed.company.clone(),
                year: parsed.year.clone(),
                text: parsed.text.clone(),
            };

            // Check for near-duplicates before saving
            match check_and_handle_duplicate(&parsed, embed_model, editor).await? {
                Some(true) => {
                    // Handled (replaced or kept existing) — don't add new
                }
                Some(false) => {
                    // User chose to edit — re-enter edit flow
                    if let Some(edited) = edit_parsed_story(&parsed, editor)? {
                        let edited_story = StorySeed {
                            company: edited.company,
                            year: edited.year,
                            text: edited.text,
                        };
                        kb::add_story_to_store(edited_story, embed_model).await?;
                        let total = kb::load_kb()?.skills.len();
                        println!("Saved to knowledge base ({} stories total)", total);
                        story_count += 1;
                    }
                }
                None => {
                    // No duplicate or user chose save-as-new
                    kb::add_story_to_store(story, embed_model).await?;
                    let total = kb::load_kb()?.skills.len();
                    println!("Saved to knowledge base ({} stories total)", total);
                    story_count += 1;
                }
            }

            let response = editor.read_line(
                &format!("\nAdd another story for {}? [y/N] ", skill.title)
            )?;
            let is_yes = response
                .as_deref()
                .is_some_and(|r| r.eq_ignore_ascii_case("y"));
            if !is_yes {
                break;
            }
        }
        if story_count > 1 {
            println!("Added {} stories for {}.", story_count, skill.title);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hiring_manager::SkillNeed;

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

    fn test_skill() -> SkillNeed {
        SkillNeed {
            title: "Rust".to_string(),
            description: "Systems programming".to_string(),
            need: 8,
            suitability: 4,
            skill_description: "Rust language".to_string(),
            justification: "Some experience".to_string(),
        }
    }

    #[test]
    fn fallback_question_adjacent_uses_title() {
        let skill = test_skill();
        let question = fallback_question(&skill, &[], &NextAction::AskAdjacent);
        assert!(question.contains("Rust"));
        assert!(question.contains("adjacent"));
    }

    #[test]
    fn fallback_question_followup_without_missing() {
        let skill = test_skill();
        let question = fallback_question(&skill, &[], &NextAction::AskFollowup);
        assert_eq!(question, "Could you add a bit more detail to your story?");
    }

    #[test]
    fn fallback_question_followup_with_missing() {
        let skill = test_skill();
        let missing = vec!["company".to_string(), "year".to_string()];
        let question = fallback_question(&skill, &missing, &NextAction::AskFollowup);
        assert_eq!(question, "Could you add details for company, year?");
    }

    #[test]
    fn parse_command_index_valid() {
        assert_eq!(parse_command_index("/use 1", "/use", 3), Some(0));
        assert_eq!(parse_command_index("/use 3", "/use", 3), Some(2));
        assert_eq!(parse_command_index("/edit 2", "/edit", 5), Some(1));
    }

    #[test]
    fn parse_command_index_out_of_range() {
        assert_eq!(parse_command_index("/use 0", "/use", 3), None);
        assert_eq!(parse_command_index("/use 4", "/use", 3), None);
    }

    #[test]
    fn parse_command_index_invalid() {
        assert_eq!(parse_command_index("/use abc", "/use", 3), None);
        assert_eq!(parse_command_index("not a command", "/use", 3), None);
    }

    #[test]
    fn split_trailing_command_done_after_text() {
        let input = "I built a simulator in Rust\n/done";
        let (before, cmd) = split_trailing_command(input).unwrap();
        assert_eq!(before, "I built a simulator in Rust");
        assert_eq!(cmd, "/done");
    }

    #[test]
    fn split_trailing_command_done_after_multiline_text() {
        let input = "Line one\nLine two\n/done";
        let (before, cmd) = split_trailing_command(input).unwrap();
        assert_eq!(before, "Line one\nLine two");
        assert_eq!(cmd, "/done");
    }

    #[test]
    fn split_trailing_command_done_with_trailing_blank_lines() {
        let input = "Some story text\n/done\n\n";
        let (before, cmd) = split_trailing_command(input).unwrap();
        assert_eq!(before, "Some story text");
        assert_eq!(cmd, "/done");
    }

    #[test]
    fn split_trailing_command_no_command() {
        let input = "Just some text\nwith multiple lines";
        assert!(split_trailing_command(input).is_none());
    }

    #[test]
    fn split_trailing_command_only_command() {
        let input = "/done";
        let (before, cmd) = split_trailing_command(input).unwrap();
        assert_eq!(before, "");
        assert_eq!(cmd, "/done");
    }

    #[test]
    fn split_trailing_command_use_after_text() {
        let input = "Some text here\n/use 2";
        let (before, cmd) = split_trailing_command(input).unwrap();
        assert_eq!(before, "Some text here");
        assert_eq!(cmd, "/use 2");
    }
}
