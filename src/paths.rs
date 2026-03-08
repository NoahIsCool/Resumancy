use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};

const APP_NAME: &str = "pipelines";

/// Return the data directory for persistent storage (KB, templates).
///
/// Priority: `PIPELINES_DATA_DIR` env var, then platform XDG/standard dir.
pub fn data_dir() -> anyhow::Result<PathBuf> {
    if let Ok(dir) = env::var("PIPELINES_DATA_DIR") {
        return Ok(PathBuf::from(dir));
    }
    dirs::data_dir()
        .map(|d| d.join(APP_NAME))
        .ok_or_else(|| anyhow!("could not determine data directory"))
}

/// Return the cache directory for LLM response caching.
///
/// When `PIPELINES_DATA_DIR` is set, cache goes under `$PIPELINES_DATA_DIR/cache`.
/// Otherwise uses the platform cache directory.
pub fn cache_dir() -> anyhow::Result<PathBuf> {
    if let Ok(dir) = env::var("PIPELINES_DATA_DIR") {
        return Ok(PathBuf::from(dir).join("cache"));
    }
    dirs::cache_dir()
        .map(|d| d.join(APP_NAME))
        .ok_or_else(|| anyhow!("could not determine cache directory"))
}

/// Return the default knowledge base path: `<data_dir>/user_skills.json`.
pub fn kb_path() -> anyhow::Result<PathBuf> {
    Ok(data_dir()?.join("user_skills.json"))
}

/// Create the directory (and parents) if it doesn't exist.
pub fn ensure_dir(path: &Path) -> anyhow::Result<()> {
    if !path.exists() {
        fs::create_dir_all(path)
            .with_context(|| format!("failed to create directory: {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
                Some(v) => unsafe { env::set_var(key, v) },
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
                Some(v) => unsafe { env::set_var(&self.key, v) },
                None => unsafe { env::remove_var(&self.key) },
            }
        }
    }

    #[test]
    fn data_dir_uses_env_override() {
        let _g = EnvGuard::set("PIPELINES_DATA_DIR", Some("/tmp/test_data"));
        assert_eq!(data_dir().unwrap(), PathBuf::from("/tmp/test_data"));
    }

    #[test]
    fn cache_dir_uses_env_override() {
        let _g = EnvGuard::set("PIPELINES_DATA_DIR", Some("/tmp/test_data"));
        assert_eq!(cache_dir().unwrap(), PathBuf::from("/tmp/test_data/cache"));
    }

    #[test]
    fn kb_path_uses_env_override() {
        let _g = EnvGuard::set("PIPELINES_DATA_DIR", Some("/tmp/test_data"));
        assert_eq!(
            kb_path().unwrap(),
            PathBuf::from("/tmp/test_data/user_skills.json")
        );
    }

    #[test]
    fn data_dir_falls_back_to_platform() {
        let _g = EnvGuard::set("PIPELINES_DATA_DIR", None);
        let dir = data_dir().unwrap();
        assert!(dir.ends_with("pipelines"));
    }

    #[test]
    fn cache_dir_falls_back_to_platform() {
        let _g = EnvGuard::set("PIPELINES_DATA_DIR", None);
        let dir = cache_dir().unwrap();
        assert!(dir.ends_with("pipelines"));
    }

    #[test]
    fn ensure_dir_creates_nested() {
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("a").join("b").join("c");
        ensure_dir(&nested).unwrap();
        assert!(nested.exists());
    }

    #[test]
    fn ensure_dir_noop_for_existing() {
        let tmp = tempfile::tempdir().unwrap();
        ensure_dir(tmp.path()).unwrap(); // should not error
    }
}
