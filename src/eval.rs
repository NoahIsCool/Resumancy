use anyhow::Result;
use rig::completion::CompletionModel;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::llm::{prompt_structured, CacheConfig, Provider};
use crate::prompts;

const DEFAULT_MIN_SCORE: u8 = 7;
const MAX_EVAL_ROUNDS: usize = 2;

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[schemars(deny_unknown_fields)]
pub struct ResumeEvaluation {
    pub overall_score: u8,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub suggestions: Vec<String>,
}

pub async fn evaluate_resume<M: CompletionModel + Clone>(
    resume_latex: &str,
    job_text: &str,
    model: &M,
    provider: Provider,
    cache: Option<&CacheConfig>,
) -> Result<ResumeEvaluation> {
    let prompt = format!(
        "JOB POSTING:\n{}\n\nGENERATED RESUME (LaTeX):\n{}\n",
        job_text, resume_latex
    );

    let result = prompt_structured(
        model,
        prompts::EVAL_PREAMBLE,
        &prompt,
        "resume_evaluation",
        provider,
        cache,
    )
    .await?;

    Ok(result.value)
}

/// Evaluate and optionally regenerate the resume. Returns the final LaTeX and evaluation.
/// `generate_fn` is called to regenerate when the score is below threshold.
#[tracing::instrument(skip_all)]
pub async fn eval_loop<M, F, Fut>(
    initial_latex: &str,
    job_text: &str,
    model: &M,
    provider: Provider,
    min_score: Option<u8>,
    cache: Option<&CacheConfig>,
    mut generate_fn: F,
) -> Result<(String, ResumeEvaluation)>
where
    M: CompletionModel + Clone,
    F: FnMut(&ResumeEvaluation) -> Fut,
    Fut: std::future::Future<Output = Result<String>>,
{
    let threshold = min_score.unwrap_or(DEFAULT_MIN_SCORE);
    let mut latex = initial_latex.to_string();

    for round in 0..=MAX_EVAL_ROUNDS {
        let eval = evaluate_resume(&latex, job_text, model, provider, cache).await?;

        if eval.overall_score >= threshold || round == MAX_EVAL_ROUNDS {
            return Ok((latex, eval));
        }

        eprintln!(
            "Evaluation score {}/9 (threshold: {}). Regenerating (attempt {}/{})...",
            eval.overall_score,
            threshold,
            round + 1,
            MAX_EVAL_ROUNDS,
        );

        latex = generate_fn(&eval).await?;
    }

    // Unreachable due to loop structure, but satisfy compiler
    let eval = evaluate_resume(&latex, job_text, model, provider, cache).await?;
    Ok((latex, eval))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_min_score_is_7() {
        assert_eq!(DEFAULT_MIN_SCORE, 7);
    }

    #[test]
    fn max_eval_rounds_is_2() {
        assert_eq!(MAX_EVAL_ROUNDS, 2);
    }

    #[test]
    fn evaluation_struct_deserializes() {
        let json = r#"{
            "overall_score": 8,
            "strengths": ["Good skills match"],
            "weaknesses": ["Missing metrics"],
            "suggestions": ["Add numbers"]
        }"#;
        let eval: ResumeEvaluation = serde_json::from_str(json).expect("parse");
        assert_eq!(eval.overall_score, 8);
        assert_eq!(eval.strengths.len(), 1);
        assert_eq!(eval.weaknesses.len(), 1);
        assert_eq!(eval.suggestions.len(), 1);
    }
}
