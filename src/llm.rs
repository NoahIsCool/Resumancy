use anyhow::{anyhow, Context, Result};
use rig::completion::{AssistantContent, CompletionModel};
use rig::OneOrMany;
use schemars::{schema_for, JsonSchema};
use serde::de::DeserializeOwned;

pub const MODEL_NAME: &str = "gpt-5.2";
pub const EMBEDDING_MODEL_NAME: &str = "text-embedding-ada-002";

fn text_from_choice(choice: OneOrMany<AssistantContent>) -> Result<String> {
    let parts = choice
        .into_iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text),
            _ => None,
        })
        .collect::<Vec<_>>();

    if parts.is_empty() {
        return Err(anyhow!("No text content in completion response"));
    }

    Ok(parts.join("\n"))
}

fn structured_output_params<T: JsonSchema>(schema_name: &str) -> Result<serde_json::Value> {
    let schema = serde_json::to_value(schema_for!(T))
        .context("failed to serialize structured output schema")?;

    Ok(serde_json::json!({
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": true
            }
        }
    }))
}

pub async fn prompt_structured<M, T>(
    model: &M,
    preamble: &str,
    prompt: &str,
    schema_name: &str,
) -> Result<T>
where
    M: CompletionModel + Clone,
    T: JsonSchema + DeserializeOwned,
{
    let params = structured_output_params::<T>(schema_name)?;
    let response = model
        .completion_request(prompt)
        .preamble(preamble.to_string())
        .additional_params(params)
        .temperature(0.0)
        .send()
        .await
        .context("structured prompt failed")?;

    let text = text_from_choice(response.choice)?;
    let parsed = serde_json::from_str::<T>(text.trim())
        .with_context(|| format!("failed to parse structured output for {schema_name}"))?;
    Ok(parsed)
}

pub async fn prompt_text<M>(model: &M, preamble: &str, prompt: &str) -> Result<String>
where
    M: CompletionModel + Clone,
{
    let response = model
        .completion_request(prompt)
        .preamble(preamble.to_string())
        .temperature(0.0)
        .send()
        .await
        .context("prompt failed")?;

    text_from_choice(response.choice)
}
