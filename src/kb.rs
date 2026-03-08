use std::fs;
use std::io;
use std::path::Path;
use anyhow::Context;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::input::InputEditor;

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
pub struct UserSkillStore {
    #[serde(default)]
    pub embedding_model: Option<String>,
    pub skills: Vec<Story>,
    #[serde(default)]
    pub user_profile: Option<UserProfile>,
}

pub fn story_document(company: &str, year: &str, text: &str) -> String {
    format!("{company} ({year}): {text}")
}

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

pub async fn retrieve_relevant_stories(
    query: &str,
    top_n: usize,
    embed_model: &impl rig::embeddings::EmbeddingModel,
) -> anyhow::Result<Vec<Story>> {
    retrieve_relevant_stories_at(&crate::paths::kb_path()?, query, top_n, embed_model).await
}

pub async fn retrieve_relevant_stories_at(
    path: &Path,
    query: &str,
    top_n: usize,
    embed_model: &impl rig::embeddings::EmbeddingModel,
) -> anyhow::Result<Vec<Story>> {
    let store = load_kb_at(path)?;
    if store.skills.is_empty() {
        return Ok(Vec::new());
    }

    let query_embedding = embed_model.embed_text(query).await?;
    let query_vec = &query_embedding.vec;

    let mut scored: Vec<(f64, &Story)> = store
        .skills
        .iter()
        .map(|story| {
            let score = cosine_similarity(query_vec, &story.vector);
            (score, story)
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_n);

    Ok(scored.into_iter().map(|(_, story)| story.clone()).collect())
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

/// Find stories whose embedding is above `threshold` similarity to `query_vec`.
/// Returns `(index, similarity, story)` triples sorted by similarity descending.
pub fn find_similar_stories<'a>(store: &'a UserSkillStore, query_vec: &[f64], threshold: f64) -> Vec<(usize, f64, &'a Story)> {
    let mut results: Vec<(usize, f64, &Story)> = store
        .skills
        .iter()
        .enumerate()
        .map(|(i, story)| {
            let sim = cosine_similarity(query_vec, &story.vector);
            (i, sim, story)
        })
        .filter(|(_, sim, _)| *sim >= threshold)
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Replace the story at `index` with new content and re-embed.
pub async fn update_story(
    index: usize,
    story: StorySeed,
    embed_model: &impl rig::embeddings::EmbeddingModel,
) -> anyhow::Result<()> {
    update_story_at(&crate::paths::kb_path()?, index, story, embed_model).await
}

pub async fn update_story_at(
    path: &Path,
    index: usize,
    story: StorySeed,
    embed_model: &impl rig::embeddings::EmbeddingModel,
) -> anyhow::Result<()> {
    let mut store = load_kb_at(path)?;
    if index >= store.skills.len() {
        return Err(anyhow::anyhow!("story index {} out of range ({})", index, store.skills.len()));
    }
    let document = story_document(&story.company, &story.year, &story.text);
    let embedding = embed_model.embed_text(&document).await?;
    store.skills[index] = Story {
        company: story.company,
        year: story.year,
        text: story.text,
        vector: embedding.vec,
    };
    save_kb_at(path, &store)?;
    Ok(())
}

/// Remove the story at `index`.
pub fn remove_story(index: usize) -> anyhow::Result<()> {
    remove_story_at(&crate::paths::kb_path()?, index)
}

pub fn remove_story_at(path: &Path, index: usize) -> anyhow::Result<()> {
    let mut store = load_kb_at(path)?;
    if index >= store.skills.len() {
        return Err(anyhow::anyhow!("story index {} out of range ({})", index, store.skills.len()));
    }
    store.skills.remove(index);
    save_kb_at(path, &store)?;
    Ok(())
}

pub async fn add_story_to_store(
    story: StorySeed,
    embed_model: &impl rig::embeddings::EmbeddingModel,
) -> anyhow::Result<()> {
    add_story_to_store_at(&crate::paths::kb_path()?, story, embed_model).await
}

pub async fn add_story_to_store_at(
    path: &Path,
    story: StorySeed,
    embed_model: &impl rig::embeddings::EmbeddingModel,
) -> anyhow::Result<()> {
    let mut store = load_kb_at(path)?;
    let document = story_document(&story.company, &story.year, &story.text);
    let embedding = embed_model.embed_text(&document).await?;

    store.skills.push(Story {
        company: story.company,
        year: story.year,
        text: story.text,
        vector: embedding.vec,
    });

    save_kb_at(path, &store)?;
    Ok(())
}

pub fn load_kb() -> anyhow::Result<UserSkillStore> {
    load_kb_at(&crate::paths::kb_path()?)
}

pub fn load_kb_at(path: &Path) -> anyhow::Result<UserSkillStore> {
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            return Ok(UserSkillStore {
                embedding_model: None,
                skills: Vec::new(),
                user_profile: None,
            })
        }
        Err(err) => return Err(err.into()),
    };

    Ok(serde_json::from_str::<UserSkillStore>(&contents)?)
}

pub fn save_kb(store: &UserSkillStore) -> anyhow::Result<()> {
    save_kb_at(&crate::paths::kb_path()?, store)
}

pub fn save_kb_at(path: &Path, store: &UserSkillStore) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(store)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create kb dir: {}", parent.display()))?;
        }
    }
    fs::write(path, json)?;
    Ok(())
}

pub fn list_story_documents() -> anyhow::Result<Vec<String>> {
    list_story_documents_at(&crate::paths::kb_path()?)
}

pub fn list_story_documents_at(path: &Path) -> anyhow::Result<Vec<String>> {
    let store = load_kb_at(path)?;
    let documents = store
        .skills
        .iter()
        .map(|story| story_document(&story.company, &story.year, &story.text))
        .collect();
    Ok(documents)
}

pub fn get_or_build_user_profile(editor: &mut InputEditor) -> anyhow::Result<UserProfile> {
    match get_user_profile()? {
        Some(profile) => Ok(profile),
        None => {
            println!("\nNo user profile found in knowledge base. Let's add it.");
            let profile = collect_user_profile(editor)?;
            set_user_profile(profile.clone())?;
            Ok(profile)
        }
    }
}

pub fn get_user_profile() -> anyhow::Result<Option<UserProfile>> {
    get_user_profile_at(&crate::paths::kb_path()?)
}

pub fn get_user_profile_at(path: &Path) -> anyhow::Result<Option<UserProfile>> {
    let store = load_kb_at(path)?;
    Ok(store.user_profile)
}

fn collect_user_profile(editor: &mut InputEditor) -> anyhow::Result<UserProfile> {
    println!("\n=== User profile ===");
    let name = editor.read_required_line("Full name: ")?;
    let location = editor.read_required_line("Location: ")?;
    let email = editor.read_required_line("Email: ")?;
    let phone = editor.read_required_line("Phone number: ")?;

    println!("\n--- Relevant links (e.g., GitHub, LinkedIn) ---");
    let mut links = Vec::new();
    loop {
        let label = editor.read_line("Link label (blank to finish): ")?;
        let Some(label) = label else { break };
        let url = editor.read_required_line(&format!("URL for {}: ", label))?;
        links.push(ProfileLink { label, url });
    }

    println!("\n--- Education ---");
    let mut education = Vec::new();
    loop {
        let degree = editor.read_line("Degree title (blank to finish): ")?;
        let Some(degree) = degree else { break };
        let graduation_date = editor.read_required_line("Graduation date: ")?;
        education.push(EducationEntry {
            degree,
            graduation_date,
        });
    }

    println!("\n--- Job history ---");
    let mut jobs = Vec::new();
    loop {
        let company = editor.read_line("Company name (blank to finish): ")?;
        let Some(company) = company else { break };
        let title = editor.read_required_line("Job title: ")?;
        let job_location = editor.read_required_line("Job location: ")?;
        let start_date = editor.read_required_line("Start date: ")?;
        let end_date = editor.read_required_line("End date (or Present): ")?;
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
    set_user_profile_at(&crate::paths::kb_path()?, profile)
}

pub fn set_user_profile_at(path: &Path, profile: UserProfile) -> anyhow::Result<()> {
    let mut store = load_kb_at(path)?;
    store.user_profile = Some(profile);
    save_kb_at(path, &store)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel};
    use std::future;

    #[derive(Clone)]
    struct DummyEmbeddingModel {
        vector: Vec<f64>,
    }

    impl EmbeddingModel for DummyEmbeddingModel {
        const MAX_DOCUMENTS: usize = 16;
        type Client = ();

        fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
            Self {
                vector: vec![0.1, 0.2, 0.3],
            }
        }

        fn ndims(&self) -> usize {
            self.vector.len()
        }

        fn embed_texts(
            &self,
            texts: impl IntoIterator<Item = String> + Send,
        ) -> impl future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + Send {
            let vec = self.vector.clone();
            let embeddings = texts
                .into_iter()
                .map(|text| Embedding {
                    document: text,
                    vec: vec.clone(),
                })
                .collect::<Vec<_>>();
            async move { Ok(embeddings) }
        }
    }

    #[test]
    fn story_document_formats() {
        let doc = story_document("Acme", "2022", "Shipped a thing");
        assert_eq!(doc, "Acme (2022): Shipped a thing");
    }

    #[test]
    fn story_id_is_deterministic_and_unique() {
        let id1 = story_id("Acme", "2022", "Did stuff");
        let id2 = story_id("Acme", "2022", "Did stuff");
        let id3 = story_id("Acme", "2023", "Did stuff");
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert!(id1.starts_with("story_"));
        assert_eq!(id1.len(), "story_".len() + 64);
    }

    #[test]
    fn load_kb_missing_file_returns_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("missing.json");
        let store = load_kb_at(&path).expect("load");
        assert!(store.skills.is_empty());
        assert!(store.user_profile.is_none());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("kb.json");
        let store = UserSkillStore {
            embedding_model: None,
            skills: vec![Story {
                company: "Acme".to_string(),
                year: "2022".to_string(),
                text: "Did stuff".to_string(),
                vector: vec![1.0, 2.0],
            }],
            user_profile: Some(UserProfile {
                name: "Ada".to_string(),
                location: "NYC".to_string(),
                email: "ada@example.com".to_string(),
                phone: "555-1234".to_string(),
                links: vec![ProfileLink {
                    label: "GitHub".to_string(),
                    url: "https://github.com/ada".to_string(),
                }],
                education: vec![EducationEntry {
                    degree: "BS CS".to_string(),
                    graduation_date: "2020".to_string(),
                }],
                jobs: vec![JobEntry {
                    company: "Acme".to_string(),
                    title: "Engineer".to_string(),
                    location: "Remote".to_string(),
                    start_date: "2020".to_string(),
                    end_date: "2022".to_string(),
                }],
            }),
        };

        save_kb_at(&path, &store).expect("save");
        let loaded = load_kb_at(&path).expect("load");
        assert_eq!(
            serde_json::to_value(store).unwrap(),
            serde_json::to_value(loaded).unwrap()
        );
    }

    #[test]
    fn save_kb_creates_parent_dirs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nested").join("kb.json");
        let store = UserSkillStore {
            embedding_model: None,
            skills: Vec::new(),
            user_profile: None,
        };
        save_kb_at(&path, &store).expect("save");
        assert!(path.exists());
    }

    #[test]
    fn list_story_documents_returns_expected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("kb.json");
        let store = UserSkillStore {
            embedding_model: None,
            skills: vec![
                Story {
                    company: "Acme".to_string(),
                    year: "2022".to_string(),
                    text: "Did stuff".to_string(),
                    vector: vec![],
                },
                Story {
                    company: "Beta".to_string(),
                    year: "2021".to_string(),
                    text: "Built thing".to_string(),
                    vector: vec![],
                },
            ],
            user_profile: None,
        };
        save_kb_at(&path, &store).expect("save");
        let docs = list_story_documents_at(&path).expect("list");
        assert_eq!(
            docs,
            vec![
                "Acme (2022): Did stuff".to_string(),
                "Beta (2021): Built thing".to_string(),
            ]
        );
    }

    #[test]
    fn set_and_get_user_profile() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("kb.json");
        save_kb_at(
            &path,
            &UserSkillStore {
                embedding_model: None,
                skills: Vec::new(),
                user_profile: None,
            },
        )
        .expect("save");

        let profile = UserProfile {
            name: "Ada".to_string(),
            location: "NYC".to_string(),
            email: "ada@example.com".to_string(),
            phone: "555-1234".to_string(),
            links: Vec::new(),
            education: Vec::new(),
            jobs: Vec::new(),
        };
        set_user_profile_at(&path, profile.clone()).expect("set");
        let loaded = get_user_profile_at(&path).expect("get");
        assert_eq!(
            serde_json::to_value(profile).unwrap(),
            serde_json::to_value(loaded.unwrap()).unwrap()
        );
    }

    #[tokio::test]
    async fn add_story_to_store_adds_embedding() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("kb.json");
        save_kb_at(
            &path,
            &UserSkillStore {
                embedding_model: None,
                skills: Vec::new(),
                user_profile: None,
            },
        )
        .expect("save");

        let model = DummyEmbeddingModel {
            vector: vec![0.5, 0.6],
        };
        let story = StorySeed {
            company: "Acme".to_string(),
            year: "2022".to_string(),
            text: "Did stuff".to_string(),
        };

        add_story_to_store_at(&path, story, &model)
            .await
            .expect("add");
        let loaded = load_kb_at(&path).expect("load");
        assert_eq!(loaded.skills.len(), 1);
        assert_eq!(loaded.skills[0].vector, vec![0.5, 0.6]);
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let score = cosine_similarity(&a, &a);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let score = cosine_similarity(&a, &b);
        assert!(score.abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_empty_returns_zero() {
        let score = cosine_similarity(&[], &[]);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn cosine_similarity_mismatched_lengths_returns_zero() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        let score = cosine_similarity(&a, &b);
        assert_eq!(score, 0.0);
    }

    #[tokio::test]
    async fn retrieve_relevant_stories_orders_by_similarity() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("kb.json");
        let store = UserSkillStore {
            embedding_model: None,
            skills: vec![
                Story {
                    company: "Far".to_string(),
                    year: "2020".to_string(),
                    text: "Unrelated".to_string(),
                    vector: vec![0.0, 0.0, 1.0],
                },
                Story {
                    company: "Close".to_string(),
                    year: "2021".to_string(),
                    text: "Relevant".to_string(),
                    vector: vec![1.0, 0.0, 0.0],
                },
            ],
            user_profile: None,
        };
        save_kb_at(&path, &store).expect("save");

        // Dummy model returns [1.0, 0.0, 0.0] — should match "Close" best
        let model = DummyEmbeddingModel {
            vector: vec![1.0, 0.0, 0.0],
        };
        let results = retrieve_relevant_stories_at(&path, "query", 2, &model)
            .await
            .expect("retrieve");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].company, "Close");
        assert_eq!(results[1].company, "Far");
    }

    #[tokio::test]
    async fn retrieve_relevant_stories_respects_top_n() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("kb.json");
        let store = UserSkillStore {
            embedding_model: None,
            skills: vec![
                Story { company: "A".into(), year: "2020".into(), text: "a".into(), vector: vec![1.0, 0.0] },
                Story { company: "B".into(), year: "2021".into(), text: "b".into(), vector: vec![0.9, 0.1] },
                Story { company: "C".into(), year: "2022".into(), text: "c".into(), vector: vec![0.0, 1.0] },
            ],
            user_profile: None,
        };
        save_kb_at(&path, &store).expect("save");

        let model = DummyEmbeddingModel { vector: vec![1.0, 0.0] };
        let results = retrieve_relevant_stories_at(&path, "q", 1, &model)
            .await
            .expect("retrieve");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].company, "A");
    }

    #[test]
    fn backward_compat_loads_without_embedding_model_field() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("kb.json");
        // JSON without embedding_model field — simulates old format
        let json = r#"{"skills":[],"user_profile":null}"#;
        fs::write(&path, json).expect("write");
        let store = load_kb_at(&path).expect("load");
        assert!(store.embedding_model.is_none());
        assert!(store.skills.is_empty());
    }

    #[test]
    fn embedding_model_field_roundtrips() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("kb.json");
        let store = UserSkillStore {
            embedding_model: Some("text-embedding-3-small".to_string()),
            skills: Vec::new(),
            user_profile: None,
        };
        save_kb_at(&path, &store).expect("save");
        let loaded = load_kb_at(&path).expect("load");
        assert_eq!(loaded.embedding_model.as_deref(), Some("text-embedding-3-small"));
    }
}
