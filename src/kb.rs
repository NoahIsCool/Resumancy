use std::{fs, io};
use std::io::Write;
use std::path::Path;
use anyhow::{anyhow, Context};
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

fn read_required_line(prompt: &str) -> anyhow::Result<String> {
    loop {
        print!("{prompt}");
        io::stdout().flush().ok();
        let mut line = String::new();
        let bytes = io::stdin().read_line(&mut line)?;
        if bytes == 0 {
            return Err(anyhow!("stdin closed while reading input"));
        }
        let value = line.trim_end_matches(&['\r', '\n'][..]).to_string();
        if !value.trim().is_empty() {
            return Ok(value);
        }
        println!("Value cannot be empty.");
    }
}

fn read_optional_line(prompt: &str) -> anyhow::Result<Option<String>> {
    print!("{prompt}");
    io::stdout().flush().ok();
    let mut line = String::new();
    let bytes = io::stdin().read_line(&mut line)?;
    if bytes == 0 {
        return Ok(None);
    }
    let value = line.trim_end_matches(&['\r', '\n'][..]).to_string();
    if value.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(value))
}

pub async fn add_story_to_store(
    story: StorySeed,
    embed_model: &impl rig::embeddings::EmbeddingModel,
) -> anyhow::Result<()> {
    let mut store = load_kb()?;
    let document = story_document(&story.company, &story.year, &story.text);
    let embedding = embed_model.embed_text(&document).await?;

    store.skills.push(Story {
        company: story.company,
        year: story.year,
        text: story.text,
        vector: embedding.vec,
    });

    save_kb(&store)?;
    Ok(())
}

pub fn load_kb() -> anyhow::Result<UserSkillStore> {
    let contents = match fs::read_to_string(DEFAULT_KB_PATH) {
        Ok(contents) => contents,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            return Ok(UserSkillStore {
                skills: Vec::new(),
                user_profile: None,
            })
        }
        Err(err) => return Err(err.into()),
    };

    Ok(serde_json::from_str::<UserSkillStore>(&contents)?)
}

pub fn save_kb(store: &UserSkillStore) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(store)?;
    let out_path = Path::new(DEFAULT_KB_PATH);
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create kb dir: {}", parent.display()))?;
        }
    }
    fs::write(out_path, json)?;
    Ok(())
}

pub fn list_story_documents() -> anyhow::Result<Vec<String>> {
    let store = load_kb()?;
    let documents = store
        .skills
        .iter()
        .map(|story| story_document(&story.company, &story.year, &story.text))
        .collect();
    Ok(documents)
}

pub fn get_or_build_user_profile() -> anyhow::Result<UserProfile> {
    match get_user_profile()? {
        Some(profile) => Ok(profile),
        None => {
            println!("\nNo user profile found in knowledge base. Let's add it.");
            let profile = collect_user_profile()?;
            set_user_profile(profile.clone())?;
            Ok(profile)
        }
    }
}

pub fn get_user_profile() -> anyhow::Result<Option<UserProfile>> {
    let store = load_kb()?;
    Ok(store.user_profile)
}

fn collect_user_profile() -> anyhow::Result<UserProfile> {
    println!("\n=== User profile ===");
    let name = read_required_line("Full name: ")?;
    let location = read_required_line("Location: ")?;
    let email = read_required_line("Email: ")?;
    let phone = read_required_line("Phone number: ")?;

    println!("\n--- Relevant links (e.g., GitHub, LinkedIn) ---");
    let mut links = Vec::new();
    loop {
        let label = read_optional_line("Link label (blank to finish): ")?;
        let Some(label) = label else { break };
        let url = read_required_line(&format!("URL for {}: ", label))?;
        links.push(ProfileLink { label, url });
    }

    println!("\n--- Education ---");
    let mut education = Vec::new();
    loop {
        let degree = read_optional_line("Degree title (blank to finish): ")?;
        let Some(degree) = degree else { break };
        let graduation_date = read_required_line("Graduation date: ")?;
        education.push(EducationEntry {
            degree,
            graduation_date,
        });
    }

    println!("\n--- Job history ---");
    let mut jobs = Vec::new();
    loop {
        let company = read_optional_line("Company name (blank to finish): ")?;
        let Some(company) = company else { break };
        let title = read_required_line("Job title: ")?;
        let job_location = read_required_line("Job location: ")?;
        let start_date = read_required_line("Start date: ")?;
        let end_date = read_required_line("End date (or Present): ")?;
        jobs.push(JobEntry {
            company,
            title,
            location: job_location,
            start_date,
            end_date,
        });
    }

    Ok(UserProfile {
        name,
        location,
        email,
        phone,
        links,
        education,
        jobs,
    })
}

pub fn set_user_profile(profile: UserProfile) -> anyhow::Result<()> {
    let mut store = load_kb()?;
    store.user_profile = Some(profile);
    save_kb(&store)?;
    Ok(())
}
