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
pub struct ProfileLink {
    pub label: String,
    pub url: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
pub struct EducationEntry {
    pub degree: String,
    pub graduation_date: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
pub struct JobEntry {
    pub company: String,
    pub title: String,
    pub location: String,
    pub start_date: String,
    pub end_date: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
pub struct UserProfile {
    pub name: String,
    pub location: String,
    pub email: String,
    pub phone: String,
    pub links: Vec<ProfileLink>,
    pub education: Vec<EducationEntry>,
    pub jobs: Vec<JobEntry>,
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
    #[serde(default)]
    pub user_profile: Option<UserProfile>,
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
