use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const DEFAULT_KB_PATH: &str = "data/user_skills.json";

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
pub struct StorySeed {
    pub company: String,
    pub year: String,
    pub text: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
pub struct UserSkillStoreSeed {
    pub skills: Vec<StorySeed>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
pub struct Story {
    pub company: String,
    pub year: String,
    pub text: String,
    #[serde(default)]
    pub vector: Vec<f64>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
pub struct UserSkillStore {
    pub skills: Vec<Story>,
}

pub fn story_document(company: &str, year: &str, text: &str) -> String {
    format!("{company} ({year}): {text}")
}

pub fn story_id(company: &str, year: &str, text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(company.as_bytes());
    hasher.update(b"|");
    hasher.update(year.as_bytes());
    hasher.update(b"|");
    hasher.update(text.as_bytes());
    format!("story_{}", hex::encode(hasher.finalize()))
}
