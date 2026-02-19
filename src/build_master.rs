use std::{fs, io, path::Path};

use anyhow::{Context, Result};
use pipelines::kb::{story_document, story_id, Story, StorySeed, UserSkillStore, DEFAULT_KB_PATH};
use pipelines::llm::EMBEDDING_MODEL_NAME;
use rig::{
    embeddings::Embedding,
    providers::openai,
    vector_store::in_memory_store::InMemoryVectorStore,
    OneOrMany,
};
use rig::client::{EmbeddingsClient, ProviderClient};
use rig::vector_store::request::VectorSearchRequest;
use rig::vector_store::VectorStoreIndex;

fn load_kb() -> Result<UserSkillStore> {
    let contents = match fs::read_to_string(DEFAULT_KB_PATH) {
        Ok(contents) => contents,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            return Ok(UserSkillStore { skills: Vec::new() })
        }
        Err(err) => return Err(err.into()),
    };

    Ok(serde_json::from_str::<UserSkillStore>(&contents)?)
}

fn save_kb(store: &UserSkillStore) -> Result<()> {
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

async fn ensure_vectors(
    store: &mut UserSkillStore,
    embed_model: &impl rig::embeddings::EmbeddingModel,
) -> Result<bool> {
    let mut updated = false;
    for story in &mut store.skills {
        if story.vector.is_empty() {
            let document = story_document(&story.company, &story.year, &story.text);
            let embedding = embed_model.embed_text(&document).await?;
            story.vector = embedding.vec;
            updated = true;
        }
    }
    Ok(updated)
}

pub async fn add_story_to_store(
    story: StorySeed,
    embed_model: &impl rig::embeddings::EmbeddingModel,
) -> Result<()> {
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

pub async fn get_skills(job_text: String) -> Result<Vec<(f64, String, String)>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("rig=trace"))
        .init();

    let mut store = load_kb()?;
    if store.skills.is_empty() {
        return Ok(Vec::new());
    }

    let openai_client = openai::Client::from_env();
    let embed_model = openai_client.embedding_model(EMBEDDING_MODEL_NAME);

    let updated = ensure_vectors(&mut store, &embed_model).await?;
    if updated {
        save_kb(&store)?;
    }

    let documents = store.skills.iter().map(|story| {
        let document = story_document(&story.company, &story.year, &story.text);
        (
            story_id(&story.company, &story.year, &story.text),
            document.clone(),
            OneOrMany::one(Embedding {
                document,
                vec: story.vector.clone(),
            }),
        )
    });

    let vector_store: InMemoryVectorStore<String> =
        InMemoryVectorStore::from_documents_with_ids(documents);

    let index = vector_store.index(embed_model);

    let req = VectorSearchRequest::builder()
        .query(&job_text)
        .threshold(0.8)
        .samples(20)
        .build()
        .expect("valid vector search request");

    let matches = index.top_n::<String>(req).await?;
    Ok(matches)
}

pub fn list_story_documents() -> Result<Vec<String>> {
    let store = load_kb()?;
    let documents = store
        .skills
        .iter()
        .map(|story| story_document(&story.company, &story.year, &story.text))
        .collect();
    Ok(documents)
}

#[cfg(test)]
mod tests {
    use super::get_skills;
    use anyhow::Result;

    #[tokio::test]
    async fn basic() -> Result<()> {
        let job_text: &str = r#"JOB POSTING (extracted text, may include noise):
Boston Dynamics Senior Staff Perception/ML Research Engineer Apply locations Waltham Office (POST) time type Full time posted on Posted 30+ Days Ago job requisition id R2038 As a Senior Staff Perception/ML Research Engineer on the Perception and Safety R&D Team, you will join a small cross-functional group developing robotic perception technologies that will enable our robots to operate safely around people. Every day you will help research, design, and build machine learning-based perception models and algorithms to run on our robots. Your work will enable our robots to understand their environment and recognize humans. You will help integrate your algorithms into embedded systems intended to make our robots safe and reactive. In this role you will chart a path by combining the best of ML with modern safety concepts and robot behavior, ultimately creating novel solutions to one of the most important problems in robotics. If you are creative, thrive in a small team environment, and passionate about a world where humans and robots truly work together - come join us! How you will make an impact: Help build the systems that allow our robots to operate safely around people. Develop datasets, metrics, and validation plans for ML models. Build, validate, and deploy ML models to detect hazards, humans, and other environmental features. Integrate these models onto our robots' embedded systems to collect data and evaluate performance. Work to improve model accuracy and run-time performance of models on specific hardware. Lead cross-functional technical efforts involving interdisciplinary efforts to develop robotic systems. Work closely with a small team to design and prototype new payloads, platforms, and product features which create safety features for our robots. We are looking for: 7+ years of experience working with perception sensor data, including stereo, LiDAR, radar, ToF, or IR data. 5+ years of experience applying ML to perception problems, ideally on embedded systems. Deep knowledge of state of the art in related areas including human detection, autonomous vehicle and driver assist systems, and robot safety. Experience developing and deploying ML-based perception software for time-sensitive control systems, such as robotics. Experience developing specifications for perception systems from high-level product requirements. Experience with the full lifecycle of deep learning development, including network design, data management, training, evaluation, hyperparameter search, deployment, and validation. Strong communication skills, including ability to author technical documentation and deliver presentations on technical topics. History of leading cross-functional technical efforts through planning, technical requirement development, and interdisciplinary collaboration. History of working in small, interdisciplinary teams. We are interested in every qualified candidate who is eligible to work in the United States. However, we are not able to sponsor visas for this position. The pay range for this position is between $173,732 - $238,882 annually. Base pay will depend on multiple individualized factors including, but not limited to internal equity, job related knowledge, skills and experience. This range represents a good faith estimate of compensation at the time of posting. Boston Dynamics offers a generous Benefits package including medical, dental vision, 401(k), paid time off and a annual bonus structure. Additional details regarding these benefit plans will be provided if an employee receives an offer for employment.
"#;

        let skills = get_skills(job_text.into()).await?;
        for (score, id, doc) in skills {
            println!("score={:.4} id={}", score, id);
            println!("doc={}\n", doc);
        }
        Ok(())
    }
}
