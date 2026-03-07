use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const DEFAULT_CACHE_DIR: &str = "data/cache";

#[derive(Debug, Serialize, Deserialize)]
struct CacheEntry {
    text: String,
    created_at: u64,
}

pub fn cache_key(
    provider: &str,
    model: &str,
    preamble: &str,
    prompt: &str,
    temperature: f64,
    schema_name: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(provider.as_bytes());
    hasher.update(b"|");
    hasher.update(model.as_bytes());
    hasher.update(b"|");
    hasher.update(preamble.as_bytes());
    hasher.update(b"|");
    hasher.update(prompt.as_bytes());
    hasher.update(b"|");
    hasher.update(temperature.to_bits().to_le_bytes());
    if let Some(schema) = schema_name {
        hasher.update(b"|");
        hasher.update(schema.as_bytes());
    }
    hex::encode(hasher.finalize())
}

fn default_cache_dir() -> PathBuf {
    Path::new(DEFAULT_CACHE_DIR).to_path_buf()
}

pub fn get_cached(key: &str) -> Result<Option<String>> {
    get_cached_at(&default_cache_dir(), key)
}

pub fn get_cached_at(cache_dir: &Path, key: &str) -> Result<Option<String>> {
    let path = cache_dir.join(format!("{}.json", key));
    match fs::read_to_string(&path) {
        Ok(contents) => {
            let entry: CacheEntry = serde_json::from_str(&contents)
                .with_context(|| format!("failed to parse cache entry: {}", path.display()))?;
            Ok(Some(entry.text))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub fn set_cached(key: &str, text: &str) -> Result<()> {
    set_cached_at(&default_cache_dir(), key, text)
}

pub fn set_cached_at(cache_dir: &Path, key: &str, text: &str) -> Result<()> {
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("failed to create cache dir: {}", cache_dir.display()))?;
    let entry = CacheEntry {
        text: text.to_string(),
        created_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };
    let json = serde_json::to_string_pretty(&entry)?;
    let path = cache_dir.join(format!("{}.json", key));
    fs::write(&path, json)
        .with_context(|| format!("failed to write cache entry: {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_is_deterministic() {
        let k1 = cache_key("claude", "sonnet", "pre", "prompt", 0.4, None);
        let k2 = cache_key("claude", "sonnet", "pre", "prompt", 0.4, None);
        assert_eq!(k1, k2);
    }

    #[test]
    fn cache_key_varies_with_inputs() {
        let k1 = cache_key("claude", "sonnet", "pre", "prompt1", 0.4, None);
        let k2 = cache_key("claude", "sonnet", "pre", "prompt2", 0.4, None);
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_includes_schema() {
        let k1 = cache_key("claude", "sonnet", "pre", "prompt", 0.0, None);
        let k2 = cache_key("claude", "sonnet", "pre", "prompt", 0.0, Some("schema"));
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_varies_with_temperature() {
        let k1 = cache_key("claude", "sonnet", "pre", "prompt", 0.0, None);
        let k2 = cache_key("claude", "sonnet", "pre", "prompt", 0.5, None);
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_miss_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let result = get_cached_at(dir.path(), "nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn cache_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        set_cached_at(dir.path(), "test_key", "hello world").unwrap();
        let result = get_cached_at(dir.path(), "test_key").unwrap();
        assert_eq!(result, Some("hello world".to_string()));
    }

    #[test]
    fn cache_overwrites_existing() {
        let dir = tempfile::tempdir().unwrap();
        set_cached_at(dir.path(), "key", "first").unwrap();
        set_cached_at(dir.path(), "key", "second").unwrap();
        let result = get_cached_at(dir.path(), "key").unwrap();
        assert_eq!(result, Some("second".to_string()));
    }

    #[test]
    fn cache_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("nested").join("cache");
        set_cached_at(&nested, "key", "value").unwrap();
        let result = get_cached_at(&nested, "key").unwrap();
        assert_eq!(result, Some("value".to_string()));
    }
}
