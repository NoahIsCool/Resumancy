use anyhow::{anyhow, Context, Result};
use rig::providers::openai;
use rig::completion::{AssistantContent, CompletionModel};
use rig::OneOrMany;
use schemars::{schema_for, JsonSchema};
use serde::de::DeserializeOwned;
use std::{env, time::Duration};

pub const MODEL_NAME: &str = "gpt-5.2";
pub const EMBEDDING_MODEL_NAME: &str = "text-embedding-ada-002";
const DEFAULT_OPENAI_TIMEOUT_SECS: u64 = 120;
const DEFAULT_OPENAI_CONNECT_TIMEOUT_SECS: u64 = 10;

fn env_timeout_secs(var: &str, default: u64) -> Result<u64> {
    match env::var(var) {
        Ok(value) => value
            .trim()
            .parse()
            .with_context(|| format!("{var} must be an integer number of seconds")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(err) => Err(anyhow!("failed to read {var}: {err}")),
    }
}

pub fn openai_client_from_env() -> Result<openai::Client> {
    let api_key = env::var("OPENAI_API_KEY")
        .context("OPENAI_API_KEY not set (needed for OpenAI requests)")?;
    if api_key.trim().is_empty() {
        return Err(anyhow!("OPENAI_API_KEY is set but empty"));
    }

    let base_url = env::var("OPENAI_BASE_URL").ok();
    let timeout_secs = env_timeout_secs("OPENAI_TIMEOUT_SECS", DEFAULT_OPENAI_TIMEOUT_SECS)?;
    let connect_timeout_secs =
        env_timeout_secs("OPENAI_CONNECT_TIMEOUT_SECS", DEFAULT_OPENAI_CONNECT_TIMEOUT_SECS)?;

    let http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .connect_timeout(Duration::from_secs(connect_timeout_secs))
        .build()
        .context("failed to build HTTP client")?;

    let mut builder = openai::Client::<reqwest::Client>::builder()
        .api_key(api_key)
        .http_client(http_client);
    if let Some(base_url) = base_url {
        builder = builder.base_url(base_url);
    }

    builder.build().context("failed to build OpenAI client")
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use rig::completion::{
        AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
        Usage,
    };
    use rig::message::{Reasoning, Text};
    use rig::streaming::StreamingCompletionResponse;
    use rig::OneOrMany;
    use schemars::JsonSchema;
    use serde::Deserialize;
    use std::env;
    use std::sync::{Mutex, MutexGuard};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvGuard {
        _lock: MutexGuard<'static, ()>,
        key: String,
        original: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &str, value: Option<&str>) -> Self {
            let lock = ENV_LOCK.lock().expect("env lock");
            let original = env::var(key).ok();
            match value {
                Some(value) => unsafe { env::set_var(key, value) },
                None => unsafe { env::remove_var(key) },
            }
            Self {
                _lock: lock,
                key: key.to_string(),
                original,
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(value) => unsafe { env::set_var(&self.key, value) },
                None => unsafe { env::remove_var(&self.key) },
            }
        }
    }

    #[derive(Clone)]
    struct MockCompletionModel {
        response: OneOrMany<AssistantContent>,
    }

    impl CompletionModel for MockCompletionModel {
        type Response = ();
        type StreamingResponse = ();
        type Client = ();

        fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
            Self {
                response: OneOrMany::one(AssistantContent::Text(Text {
                    text: String::new(),
                })),
            }
        }

        fn completion(
            &self,
            _request: CompletionRequest,
        ) -> impl std::future::Future<
            Output = Result<CompletionResponse<Self::Response>, CompletionError>,
        > + Send {
            let choice = self.response.clone();
            async move {
                Ok(CompletionResponse {
                    choice,
                    usage: Usage::default(),
                    raw_response: (),
                })
            }
        }

        fn stream(
            &self,
            _request: CompletionRequest,
        ) -> impl std::future::Future<
            Output = Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>,
        > + Send {
            async { Err(CompletionError::ProviderError("streaming not supported".into())) }
        }
    }

    #[test]
    fn env_timeout_secs_defaults_when_missing() {
        let _guard = EnvGuard::set("PIPELINES_TEST_TIMEOUT", None);
        let value = env_timeout_secs("PIPELINES_TEST_TIMEOUT", 15).expect("timeout");
        assert_eq!(value, 15);
    }

    #[test]
    fn env_timeout_secs_parses_value() {
        let _guard = EnvGuard::set("PIPELINES_TEST_TIMEOUT", Some("42"));
        let value = env_timeout_secs("PIPELINES_TEST_TIMEOUT", 15).expect("timeout");
        assert_eq!(value, 42);
    }

    #[test]
    fn env_timeout_secs_rejects_invalid() {
        let _guard = EnvGuard::set("PIPELINES_TEST_TIMEOUT", Some("nope"));
        assert!(env_timeout_secs("PIPELINES_TEST_TIMEOUT", 15).is_err());
    }

    #[test]
    fn openai_client_from_env_errors_when_missing_key() {
        let _guard = EnvGuard::set("OPENAI_API_KEY", None);
        let err = openai_client_from_env().unwrap_err().to_string();
        assert!(err.contains("OPENAI_API_KEY not set"));
    }

    #[test]
    fn openai_client_from_env_errors_when_empty_key() {
        let _guard = EnvGuard::set("OPENAI_API_KEY", Some("   "));
        let err = openai_client_from_env().unwrap_err().to_string();
        assert!(err.contains("OPENAI_API_KEY is set but empty"));
    }

    #[test]
    fn openai_client_from_env_ok_with_key() {
        let _guard = EnvGuard::set("OPENAI_API_KEY", Some("test-key"));
        let client = openai_client_from_env();
        assert!(client.is_ok());
    }

    #[test]
    fn text_from_choice_joins_text_segments() {
        let choice = OneOrMany::many(vec![
            AssistantContent::Text(Text {
                text: "first".to_string(),
            }),
            AssistantContent::Text(Text {
                text: "second".to_string(),
            }),
        ])
        .expect("non-empty");
        let text = text_from_choice(choice).expect("text");
        assert_eq!(text, "first\nsecond");
    }

    #[test]
    fn text_from_choice_errors_without_text() {
        let choice = OneOrMany::one(AssistantContent::Reasoning(Reasoning::new(
            "reasoning",
        )));
        assert!(text_from_choice(choice).is_err());
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct DummySchema {
        answer: String,
    }

    #[test]
    fn structured_output_params_contains_schema_metadata() {
        let params = structured_output_params::<DummySchema>("dummy").expect("params");
        assert_eq!(params["text"]["format"]["type"], "json_schema");
        assert_eq!(params["text"]["format"]["name"], "dummy");
        assert_eq!(params["text"]["format"]["strict"], true);
        assert!(params["text"]["format"]["schema"]["properties"]["answer"].is_object());
    }

    #[tokio::test]
    async fn prompt_text_returns_text() {
        let model = MockCompletionModel {
            response: OneOrMany::one(AssistantContent::Text(Text {
                text: "hello".to_string(),
            })),
        };
        let text = prompt_text(&model, "preamble", "prompt")
            .await
            .expect("prompt");
        assert_eq!(text, "hello");
    }

    #[tokio::test]
    async fn prompt_structured_parses_json() {
        #[derive(Debug, Deserialize, JsonSchema)]
        struct Payload {
            value: String,
        }

        let model = MockCompletionModel {
            response: OneOrMany::one(AssistantContent::Text(Text {
                text: r#"{"value":"ok"}"#.to_string(),
            })),
        };

        let parsed: Payload =
            prompt_structured(&model, "preamble", "prompt", "payload")
                .await
                .expect("structured");
        assert_eq!(parsed.value, "ok");
    }

    #[tokio::test]
    async fn prompt_structured_errors_on_invalid_json() {
        #[derive(Debug, Deserialize, JsonSchema)]
        #[allow(dead_code)]
        struct Payload {
            value: String,
        }

        let model = MockCompletionModel {
            response: OneOrMany::one(AssistantContent::Text(Text {
                text: "not-json".to_string(),
            })),
        };

        let err = prompt_structured::<_, Payload>(&model, "preamble", "prompt", "payload")
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("failed to parse structured output"));
    }
}
