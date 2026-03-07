use anyhow::{anyhow, Context, Result};
use clap::ValueEnum;
use rig::completion::{AssistantContent, CompletionModel, Usage};
use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel};
use rig::OneOrMany;
use schemars::{schema_for, JsonSchema};
use serde::de::DeserializeOwned;
use std::{env, time::Duration};

use crate::cache;

/// Combine two `Usage` values by summing their token counts.
pub fn combine_usage(a: Usage, b: Usage) -> Usage {
    Usage {
        input_tokens: a.input_tokens + b.input_tokens,
        output_tokens: a.output_tokens + b.output_tokens,
        total_tokens: a.total_tokens + b.total_tokens,
    }
}

/// Dispatch a provider to a callback that receives completion + embedding models.
///
/// Usage:
/// ```ignore
/// dispatch_provider!(provider, &model_name, embed_override, |m, e| {
///     do_something(&m, &e).await
/// })
/// ```
#[macro_export]
macro_rules! dispatch_provider {
    ($provider:expr, $model_name:expr, $embed_override:expr,
     |$comp:ident, $embed:ident| $body:expr) => {
        match $provider {
            $crate::llm::Provider::Claude => {
                let client = $crate::llm::anthropic_client_from_env()?;
                let $comp = ::rig::client::CompletionClient::completion_model(&client, $model_name);
                let $embed = $crate::llm::NullEmbeddingModel;
                $body
            }
            $crate::llm::Provider::OpenAI => {
                let client = $crate::llm::openai_client_from_env()?;
                let $comp = ::rig::client::CompletionClient::completion_model(&client, $model_name);
                let embed_name = $embed_override
                    .or_else(|| $crate::llm::Provider::OpenAI.default_embedding_model().map(String::from))
                    .unwrap();
                let $embed = ::rig::client::EmbeddingsClient::embedding_model(&client, &embed_name);
                $body
            }
            $crate::llm::Provider::Gemini => {
                let client = $crate::llm::gemini_client_from_env()?;
                let $comp = ::rig::client::CompletionClient::completion_model(&client, $model_name);
                let embed_name = $embed_override
                    .or_else(|| $crate::llm::Provider::Gemini.default_embedding_model().map(String::from))
                    .unwrap();
                let $embed = ::rig::client::EmbeddingsClient::embedding_model(&client, &embed_name);
                $body
            }
            $crate::llm::Provider::Ollama => {
                let client = $crate::llm::ollama_client_from_env()?;
                let $comp = ::rig::client::CompletionClient::completion_model(&client, $model_name);
                if let Some(embed_name) = $embed_override
                    .or_else(|| $crate::llm::Provider::Ollama.default_embedding_model().map(String::from))
                {
                    let $embed = ::rig::client::EmbeddingsClient::embedding_model(&client, &embed_name);
                    $body
                } else {
                    let $embed = $crate::llm::NullEmbeddingModel;
                    $body
                }
            }
            $crate::llm::Provider::DeepSeek => {
                let client = $crate::llm::deepseek_client_from_env()?;
                let $comp = ::rig::client::CompletionClient::completion_model(&client, $model_name);
                let $embed = $crate::llm::NullEmbeddingModel;
                $body
            }
            $crate::llm::Provider::Groq => {
                let client = $crate::llm::groq_client_from_env()?;
                let $comp = ::rig::client::CompletionClient::completion_model(&client, $model_name);
                let $embed = $crate::llm::NullEmbeddingModel;
                $body
            }
            $crate::llm::Provider::XAI => {
                let client = $crate::llm::xai_client_from_env()?;
                let $comp = ::rig::client::CompletionClient::completion_model(&client, $model_name);
                let $embed = $crate::llm::NullEmbeddingModel;
                $body
            }
        }
    };
}

/// Dispatch a provider to a callback that receives only a completion model.
///
/// Simpler variant for code that doesn't need embeddings.
#[macro_export]
macro_rules! dispatch_completion {
    ($provider:expr, $model_name:expr, |$comp:ident| $body:expr) => {
        match $provider {
            $crate::llm::Provider::Claude => {
                let client = $crate::llm::anthropic_client_from_env()?;
                let $comp = ::rig::client::CompletionClient::completion_model(&client, $model_name);
                $body
            }
            $crate::llm::Provider::OpenAI => {
                let client = $crate::llm::openai_client_from_env()?;
                let $comp = ::rig::client::CompletionClient::completion_model(&client, $model_name);
                $body
            }
            $crate::llm::Provider::Gemini => {
                let client = $crate::llm::gemini_client_from_env()?;
                let $comp = ::rig::client::CompletionClient::completion_model(&client, $model_name);
                $body
            }
            $crate::llm::Provider::Ollama => {
                let client = $crate::llm::ollama_client_from_env()?;
                let $comp = ::rig::client::CompletionClient::completion_model(&client, $model_name);
                $body
            }
            $crate::llm::Provider::DeepSeek => {
                let client = $crate::llm::deepseek_client_from_env()?;
                let $comp = ::rig::client::CompletionClient::completion_model(&client, $model_name);
                $body
            }
            $crate::llm::Provider::Groq => {
                let client = $crate::llm::groq_client_from_env()?;
                let $comp = ::rig::client::CompletionClient::completion_model(&client, $model_name);
                $body
            }
            $crate::llm::Provider::XAI => {
                let client = $crate::llm::xai_client_from_env()?;
                let $comp = ::rig::client::CompletionClient::completion_model(&client, $model_name);
                $body
            }
        }
    };
}

const DEFAULT_TIMEOUT_SECS: u64 = 120;
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Provider {
    Claude,
    OpenAI,
    Gemini,
    Ollama,
    DeepSeek,
    Groq,
    XAI,
}

impl Provider {
    pub fn name(&self) -> &'static str {
        match self {
            Provider::Claude => "claude",
            Provider::OpenAI => "openai",
            Provider::Gemini => "gemini",
            Provider::Ollama => "ollama",
            Provider::DeepSeek => "deepseek",
            Provider::Groq => "groq",
            Provider::XAI => "xai",
        }
    }

    pub fn default_model(&self) -> &'static str {
        match self {
            Provider::Claude => "claude-sonnet-4-20250514",
            Provider::OpenAI => "gpt-4o",
            Provider::Gemini => "gemini-2.5-flash",
            Provider::Ollama => "llama3.2",
            Provider::DeepSeek => "deepseek-chat",
            Provider::Groq => "llama-3.2-70b-versatile",
            Provider::XAI => "grok-3",
        }
    }

    pub fn default_embedding_model(&self) -> Option<&'static str> {
        match self {
            Provider::OpenAI => Some("text-embedding-3-small"),
            Provider::Gemini => Some("text-embedding-004"),
            Provider::Ollama => Some("all-minilm"),
            _ => None,
        }
    }

    pub fn api_key_env_var(&self) -> Option<&'static str> {
        match self {
            Provider::Claude => Some("ANTHROPIC_API_KEY"),
            Provider::OpenAI => Some("OPENAI_API_KEY"),
            Provider::Gemini => Some("GEMINI_API_KEY"),
            Provider::Ollama => None,
            Provider::DeepSeek => Some("DEEPSEEK_API_KEY"),
            Provider::Groq => Some("GROQ_API_KEY"),
            Provider::XAI => Some("XAI_API_KEY"),
        }
    }

    fn structured_output_strategy(&self) -> StructuredOutputStrategy {
        match self {
            Provider::OpenAI => StructuredOutputStrategy::OpenAIResponsesApi,
            Provider::Gemini => StructuredOutputStrategy::GeminiGenerationConfig,
            Provider::Claude
            | Provider::DeepSeek
            | Provider::Groq
            | Provider::Ollama
            | Provider::XAI => StructuredOutputStrategy::PromptOnly,
        }
    }
}

enum StructuredOutputStrategy {
    OpenAIResponsesApi,
    GeminiGenerationConfig,
    PromptOnly,
}

#[derive(Clone)]
pub struct CacheConfig {
    pub provider_name: String,
    pub model_name: String,
    pub enabled: bool,
    pub max_age: Option<std::time::Duration>,
}

#[derive(Debug)]
pub struct LlmOutput<T> {
    pub value: T,
    pub usage: Usage,
}

/// An embedding model that returns empty vectors. Used when the provider
/// does not support embeddings. Stories are stored/retrieved with empty
/// vectors, so cosine similarity returns 0 for all — preserving insertion
/// order while still including everything.
#[derive(Clone)]
pub struct NullEmbeddingModel;

impl EmbeddingModel for NullEmbeddingModel {
    const MAX_DOCUMENTS: usize = 128;
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
        Self
    }

    fn ndims(&self) -> usize {
        0
    }

    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> impl std::future::Future<Output = std::result::Result<Vec<Embedding>, EmbeddingError>> + Send
    {
        let embeddings = texts
            .into_iter()
            .map(|text| Embedding {
                document: text,
                vec: Vec::new(),
            })
            .collect();
        async { Ok(embeddings) }
    }
}

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

fn require_api_key(env_var: &str) -> Result<String> {
    let key = env::var(env_var)
        .with_context(|| format!("{env_var} not set"))?;
    if key.trim().is_empty() {
        return Err(anyhow!("{env_var} is set but empty"));
    }
    Ok(key)
}

fn build_http_client() -> Result<reqwest::Client> {
    let timeout_secs = env_timeout_secs("LLM_TIMEOUT_SECS", DEFAULT_TIMEOUT_SECS)?;
    let connect_timeout_secs =
        env_timeout_secs("LLM_CONNECT_TIMEOUT_SECS", DEFAULT_CONNECT_TIMEOUT_SECS)?;
    reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .connect_timeout(Duration::from_secs(connect_timeout_secs))
        .build()
        .context("failed to build HTTP client")
}

// --- Provider client constructors ---

pub fn anthropic_client_from_env() -> Result<rig::providers::anthropic::Client> {
    let api_key = require_api_key("ANTHROPIC_API_KEY")?;
    let http_client = build_http_client()?;
    rig::providers::anthropic::Client::<reqwest::Client>::builder()
        .api_key(api_key)
        .http_client(http_client)
        .build()
        .context("failed to build Anthropic client")
}

pub fn openai_client_from_env() -> Result<rig::providers::openai::Client> {
    let api_key = require_api_key("OPENAI_API_KEY")?;
    let base_url = env::var("OPENAI_BASE_URL").ok();
    let http_client = build_http_client()?;
    let mut builder = rig::providers::openai::Client::<reqwest::Client>::builder()
        .api_key(api_key)
        .http_client(http_client);
    if let Some(base_url) = base_url {
        builder = builder.base_url(base_url);
    }
    builder.build().context("failed to build OpenAI client")
}

pub fn gemini_client_from_env() -> Result<rig::providers::gemini::Client> {
    let api_key = require_api_key("GEMINI_API_KEY")?;
    let http_client = build_http_client()?;
    rig::providers::gemini::Client::<reqwest::Client>::builder()
        .api_key(api_key)
        .http_client(http_client)
        .build()
        .context("failed to build Gemini client")
}

pub fn ollama_client_from_env() -> Result<rig::providers::ollama::Client> {
    let http_client = build_http_client()?;
    let mut builder = rig::providers::ollama::Client::<reqwest::Client>::builder()
        .api_key(rig::client::Nothing)
        .http_client(http_client);
    if let Some(base_url) = env::var("OLLAMA_API_BASE_URL").ok() {
        builder = builder.base_url(base_url);
    }
    builder.build().context("failed to build Ollama client")
}

pub fn deepseek_client_from_env() -> Result<rig::providers::deepseek::Client> {
    let api_key = require_api_key("DEEPSEEK_API_KEY")?;
    let http_client = build_http_client()?;
    rig::providers::deepseek::Client::<reqwest::Client>::builder()
        .api_key(api_key)
        .http_client(http_client)
        .build()
        .context("failed to build DeepSeek client")
}

pub fn groq_client_from_env() -> Result<rig::providers::groq::Client> {
    let api_key = require_api_key("GROQ_API_KEY")?;
    let http_client = build_http_client()?;
    rig::providers::groq::Client::<reqwest::Client>::builder()
        .api_key(api_key)
        .http_client(http_client)
        .build()
        .context("failed to build Groq client")
}

pub fn xai_client_from_env() -> Result<rig::providers::xai::Client> {
    let api_key = require_api_key("XAI_API_KEY")?;
    let http_client = build_http_client()?;
    rig::providers::xai::Client::<reqwest::Client>::builder()
        .api_key(api_key)
        .http_client(http_client)
        .build()
        .context("failed to build xAI client")
}

// --- Generic prompt helpers ---

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

fn openai_structured_params<T: JsonSchema>(schema_name: &str) -> Result<serde_json::Value> {
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

fn gemini_structured_params<T: JsonSchema>() -> Result<serde_json::Value> {
    let schema = serde_json::to_value(schema_for!(T))
        .context("failed to serialize structured output schema")?;

    Ok(serde_json::json!({
        "generation_config": {
            "response_mime_type": "application/json",
            "response_json_schema": schema
        }
    }))
}

fn json_preamble_suffix<T: JsonSchema>(schema_name: &str) -> Result<String> {
    let schema = serde_json::to_value(schema_for!(T))
        .context("failed to serialize structured output schema")?;
    let pretty = serde_json::to_string_pretty(&schema)?;
    Ok(format!(
        "\n\nIMPORTANT: You MUST respond with a single valid JSON object matching the \"{schema_name}\" schema below. \
         Output ONLY the JSON. No commentary, no markdown fences, no extra text.\n\
         Schema:\n{pretty}"
    ))
}

/// Strip markdown code fences from LLM output. Handles ```json, ```latex, and plain ``` fences.
pub fn strip_code_fences(text: &str) -> &str {
    let trimmed = text.trim();
    // Try specific language tags first, then plain fences
    for prefix in &["```json", "```latex", "```tex", "```"] {
        if let Some(inner) = trimmed.strip_prefix(prefix) {
            let inner = inner.trim_start();
            if let Some(stripped) = inner.strip_suffix("```") {
                return stripped.trim();
            }
        }
    }
    trimmed
}

fn strip_json_fences(text: &str) -> &str {
    let trimmed = text.trim();
    if let Some(inner) = trimmed.strip_prefix("```json") {
        let inner = inner.trim_start();
        if let Some(stripped) = inner.strip_suffix("```") {
            return stripped.trim();
        }
    }
    if let Some(inner) = trimmed.strip_prefix("```") {
        let inner = inner.trim_start();
        if let Some(stripped) = inner.strip_suffix("```") {
            return stripped.trim();
        }
    }
    trimmed
}

fn try_cache_get(cache: Option<&CacheConfig>, preamble: &str, prompt: &str, temperature: f64, schema_name: Option<&str>) -> Result<Option<String>> {
    if let Some(config) = cache {
        if config.enabled {
            let key = cache::cache_key(&config.provider_name, &config.model_name, preamble, prompt, temperature, schema_name);
            return cache::get_cached(&key, config.max_age);
        }
    }
    Ok(None)
}

fn try_cache_set(cache: Option<&CacheConfig>, preamble: &str, prompt: &str, temperature: f64, schema_name: Option<&str>, text: &str) -> Result<()> {
    if let Some(config) = cache {
        if config.enabled {
            let key = cache::cache_key(&config.provider_name, &config.model_name, preamble, prompt, temperature, schema_name);
            cache::set_cached(&key, text)?;
        }
    }
    Ok(())
}

#[tracing::instrument(skip_all, fields(
    llm.provider = tracing::field::Empty,
    llm.model = tracing::field::Empty,
    llm.temperature = 0.0,
    llm.input_tokens = tracing::field::Empty,
    llm.output_tokens = tracing::field::Empty,
    llm.cached = false,
))]
pub async fn prompt_structured<M, T>(
    model: &M,
    preamble: &str,
    prompt: &str,
    schema_name: &str,
    provider: Provider,
    cache: Option<&CacheConfig>,
) -> Result<LlmOutput<T>>
where
    M: CompletionModel + Clone,
    T: JsonSchema + DeserializeOwned,
{
    let span = tracing::Span::current();
    if let Some(config) = cache {
        span.record("llm.provider", config.provider_name.as_str());
        span.record("llm.model", config.model_name.as_str());
    }

    // Check cache
    if let Some(cached) = try_cache_get(cache, preamble, prompt, 0.0, Some(schema_name))? {
        span.record("llm.cached", true);
        let cleaned = strip_json_fences(&cached);
        let parsed = serde_json::from_str::<T>(cleaned)
            .with_context(|| format!("failed to parse cached structured output for {schema_name}"))?;
        return Ok(LlmOutput { value: parsed, usage: Usage::default() });
    }

    let strategy = provider.structured_output_strategy();

    let effective_preamble = match strategy {
        StructuredOutputStrategy::PromptOnly => {
            let suffix = json_preamble_suffix::<T>(schema_name)?;
            format!("{preamble}{suffix}")
        }
        _ => preamble.to_string(),
    };

    let mut builder = model
        .completion_request(prompt)
        .preamble(effective_preamble)
        .temperature(0.0);

    match strategy {
        StructuredOutputStrategy::OpenAIResponsesApi => {
            let params = openai_structured_params::<T>(schema_name)?;
            builder = builder.additional_params(params);
        }
        StructuredOutputStrategy::GeminiGenerationConfig => {
            let params = gemini_structured_params::<T>()?;
            builder = builder.additional_params(params);
        }
        StructuredOutputStrategy::PromptOnly => {}
    }

    let response = builder
        .send()
        .await
        .context("structured prompt failed")?;

    let usage = response.usage;
    span.record("llm.input_tokens", usage.input_tokens);
    span.record("llm.output_tokens", usage.output_tokens);

    let text = text_from_choice(response.choice)?;

    // Write cache (raw text, pre-parse)
    try_cache_set(cache, preamble, prompt, 0.0, Some(schema_name), &text)?;

    let cleaned = strip_json_fences(&text);
    let parsed = serde_json::from_str::<T>(cleaned)
        .with_context(|| format!("failed to parse structured output for {schema_name}"))?;
    Ok(LlmOutput { value: parsed, usage })
}

pub async fn prompt_text<M>(model: &M, preamble: &str, prompt: &str) -> Result<String>
where
    M: CompletionModel + Clone,
{
    Ok(prompt_text_with_temperature(model, preamble, prompt, 0.4, None).await?.value)
}

#[tracing::instrument(skip_all, fields(
    llm.provider = tracing::field::Empty,
    llm.model = tracing::field::Empty,
    llm.temperature = temperature,
    llm.input_tokens = tracing::field::Empty,
    llm.output_tokens = tracing::field::Empty,
    llm.cached = false,
))]
pub async fn prompt_text_with_temperature<M>(
    model: &M,
    preamble: &str,
    prompt: &str,
    temperature: f64,
    cache: Option<&CacheConfig>,
) -> Result<LlmOutput<String>>
where
    M: CompletionModel + Clone,
{
    let span = tracing::Span::current();
    if let Some(config) = cache {
        span.record("llm.provider", config.provider_name.as_str());
        span.record("llm.model", config.model_name.as_str());
    }

    // Check cache
    if let Some(cached) = try_cache_get(cache, preamble, prompt, temperature, None)? {
        span.record("llm.cached", true);
        return Ok(LlmOutput { value: cached, usage: Usage::default() });
    }

    let response = model
        .completion_request(prompt)
        .preamble(preamble.to_string())
        .temperature(temperature)
        .send()
        .await
        .context("prompt failed")?;

    let usage = response.usage;
    span.record("llm.input_tokens", usage.input_tokens);
    span.record("llm.output_tokens", usage.output_tokens);

    let text = text_from_choice(response.choice)?;

    // Write cache
    try_cache_set(cache, preamble, prompt, temperature, None, &text)?;

    Ok(LlmOutput { value: text, usage })
}

#[tracing::instrument(skip_all, fields(
    llm.provider = tracing::field::Empty,
    llm.model = tracing::field::Empty,
    llm.temperature = temperature,
    llm.cached = false,
))]
pub async fn prompt_text_streaming<M>(
    model: &M,
    preamble: &str,
    prompt: &str,
    temperature: f64,
    cache: Option<&CacheConfig>,
) -> Result<String>
where
    M: CompletionModel + Clone,
{
    let span = tracing::Span::current();
    if let Some(config) = cache {
        span.record("llm.provider", config.provider_name.as_str());
        span.record("llm.model", config.model_name.as_str());
    }

    // Check cache first — no streaming needed for hits
    if let Some(cached) = try_cache_get(cache, preamble, prompt, temperature, None)? {
        span.record("llm.cached", true);
        return Ok(cached);
    }

    use futures::StreamExt;

    let mut stream = model
        .completion_request(prompt)
        .preamble(preamble.to_string())
        .temperature(temperature)
        .stream()
        .await
        .context("streaming prompt failed")?;

    let mut accumulated = String::new();

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(rig::streaming::StreamedAssistantContent::Text(text)) => {
                eprint!("{}", text.text);
                accumulated.push_str(&text.text);
            }
            Err(e) => {
                return Err(anyhow!("streaming error: {}", e));
            }
            _ => {}
        }
    }
    eprintln!();

    // Write cache
    try_cache_set(cache, preamble, prompt, temperature, None, &accumulated)?;

    Ok(accumulated)
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
            Output = std::result::Result<CompletionResponse<Self::Response>, CompletionError>,
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
            Output = std::result::Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>,
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
    fn require_api_key_errors_when_missing() {
        let _guard = EnvGuard::set("PIPELINES_TEST_KEY", None);
        let err = require_api_key("PIPELINES_TEST_KEY").unwrap_err().to_string();
        assert!(err.contains("PIPELINES_TEST_KEY not set"));
    }

    #[test]
    fn require_api_key_errors_when_empty() {
        let _guard = EnvGuard::set("PIPELINES_TEST_KEY", Some("   "));
        let err = require_api_key("PIPELINES_TEST_KEY").unwrap_err().to_string();
        assert!(err.contains("is set but empty"));
    }

    #[test]
    fn require_api_key_ok() {
        let _guard = EnvGuard::set("PIPELINES_TEST_KEY", Some("test-key"));
        assert_eq!(require_api_key("PIPELINES_TEST_KEY").unwrap(), "test-key");
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
    fn openai_structured_params_contains_schema_metadata() {
        let params = openai_structured_params::<DummySchema>("dummy").expect("params");
        assert_eq!(params["text"]["format"]["type"], "json_schema");
        assert_eq!(params["text"]["format"]["name"], "dummy");
        assert_eq!(params["text"]["format"]["strict"], true);
        assert!(params["text"]["format"]["schema"]["properties"]["answer"].is_object());
    }

    #[test]
    fn gemini_structured_params_shape() {
        let params = gemini_structured_params::<DummySchema>().expect("params");
        assert_eq!(params["generation_config"]["response_mime_type"], "application/json");
        assert!(params["generation_config"]["response_json_schema"].is_object());
    }

    #[test]
    fn strip_json_fences_removes_json_fence() {
        assert_eq!(strip_json_fences("```json\n{\"a\":1}\n```"), "{\"a\":1}");
    }

    #[test]
    fn strip_json_fences_removes_plain_fence() {
        assert_eq!(strip_json_fences("```\n{\"a\":1}\n```"), "{\"a\":1}");
    }

    #[test]
    fn strip_json_fences_preserves_plain_json() {
        assert_eq!(strip_json_fences("{\"a\":1}"), "{\"a\":1}");
    }

    #[test]
    fn strip_code_fences_removes_latex_fence() {
        let input = "```latex\n\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}\n```";
        assert_eq!(
            strip_code_fences(input),
            "\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}"
        );
    }

    #[test]
    fn strip_code_fences_removes_tex_fence() {
        assert_eq!(
            strip_code_fences("```tex\n\\section{Hi}\n```"),
            "\\section{Hi}"
        );
    }

    #[test]
    fn strip_code_fences_removes_plain_fence() {
        assert_eq!(strip_code_fences("```\nsome content\n```"), "some content");
    }

    #[test]
    fn strip_code_fences_preserves_unfenced() {
        let input = "\\documentclass{article}";
        assert_eq!(strip_code_fences(input), input);
    }

    #[test]
    fn prompt_only_providers_strategy() {
        for provider in [Provider::Claude, Provider::DeepSeek, Provider::Groq, Provider::Ollama, Provider::XAI] {
            assert!(matches!(
                provider.structured_output_strategy(),
                StructuredOutputStrategy::PromptOnly
            ));
        }
    }

    #[test]
    fn openai_uses_responses_api_strategy() {
        assert!(matches!(
            Provider::OpenAI.structured_output_strategy(),
            StructuredOutputStrategy::OpenAIResponsesApi
        ));
    }

    #[test]
    fn gemini_uses_generation_config_strategy() {
        assert!(matches!(
            Provider::Gemini.structured_output_strategy(),
            StructuredOutputStrategy::GeminiGenerationConfig
        ));
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

        let result: LlmOutput<Payload> =
            prompt_structured(&model, "preamble", "prompt", "payload", Provider::OpenAI, None)
                .await
                .expect("structured");
        assert_eq!(result.value.value, "ok");
    }

    #[tokio::test]
    async fn prompt_structured_parses_json_with_fences() {
        #[derive(Debug, Deserialize, JsonSchema)]
        struct Payload {
            value: String,
        }

        let model = MockCompletionModel {
            response: OneOrMany::one(AssistantContent::Text(Text {
                text: "```json\n{\"value\":\"fenced\"}\n```".to_string(),
            })),
        };

        let result: LlmOutput<Payload> =
            prompt_structured(&model, "preamble", "prompt", "payload", Provider::Claude, None)
                .await
                .expect("structured");
        assert_eq!(result.value.value, "fenced");
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

        let err = prompt_structured::<_, Payload>(&model, "preamble", "prompt", "payload", Provider::Claude, None)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("failed to parse structured output"));
    }

    #[tokio::test]
    async fn null_embedding_model_returns_empty_vectors() {
        let model = NullEmbeddingModel;
        let embedding = model.embed_text("test").await.unwrap();
        assert!(embedding.vec.is_empty());
        assert_eq!(embedding.document, "test");
    }

    #[test]
    fn provider_default_model_names() {
        assert!(!Provider::Claude.default_model().is_empty());
        assert!(!Provider::OpenAI.default_model().is_empty());
        assert!(!Provider::Gemini.default_model().is_empty());
    }

    #[test]
    fn provider_embedding_support() {
        assert!(Provider::OpenAI.default_embedding_model().is_some());
        assert!(Provider::Gemini.default_embedding_model().is_some());
        assert!(Provider::Ollama.default_embedding_model().is_some());
        assert!(Provider::Claude.default_embedding_model().is_none());
        assert!(Provider::DeepSeek.default_embedding_model().is_none());
    }

    #[test]
    fn provider_name_strings() {
        assert_eq!(Provider::Claude.name(), "claude");
        assert_eq!(Provider::OpenAI.name(), "openai");
        assert_eq!(Provider::Gemini.name(), "gemini");
        assert_eq!(Provider::Ollama.name(), "ollama");
        assert_eq!(Provider::DeepSeek.name(), "deepseek");
        assert_eq!(Provider::Groq.name(), "groq");
        assert_eq!(Provider::XAI.name(), "xai");
    }

    #[tokio::test]
    async fn prompt_text_with_cache_hit() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("cache");
        let key = cache::cache_key("test", "test-model", "preamble", "prompt", 0.4, None);
        cache::set_cached_at(&cache_dir, &key, "cached response").unwrap();

        // Override the default cache dir for this test by directly testing try_cache_get
        let result = cache::get_cached_at(&cache_dir, &key, None).unwrap();
        assert_eq!(result, Some("cached response".to_string()));
    }

    #[test]
    fn cache_config_clone() {
        let config = CacheConfig {
            provider_name: "claude".to_string(),
            model_name: "sonnet".to_string(),
            enabled: true,
            max_age: None,
        };
        let cloned = config.clone();
        assert_eq!(cloned.provider_name, "claude");
        assert_eq!(cloned.model_name, "sonnet");
        assert!(cloned.enabled);
    }
}
