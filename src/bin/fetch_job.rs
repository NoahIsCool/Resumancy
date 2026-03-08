use std::fs;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rig::completion::CompletionModel;
use pipelines::llm::{prompt_text_with_temperature, CacheConfig, Provider};

const SKIP_TAGS: &[&str] = &[
    "script", "style", "nav", "header", "footer", "aside",
    "iframe", "form", "svg", "noscript",
];

/// Fetch a job posting URL and extract the job description as clean markdown.
///
/// If no output path is given, derives one from the extracted company name and
/// job title: `<company>/<job title>.md`
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// LLM provider to use
    #[arg(long, value_enum, env = "LLM_PROVIDER", default_value_t = Provider::Claude)]
    provider: Provider,

    /// Model name override
    #[arg(long, env = "LLM_MODEL")]
    model: Option<String>,

    /// Disable response caching
    #[arg(long, default_value_t = false)]
    no_cache: bool,

    /// Cache TTL in seconds (entries older than this are treated as misses)
    #[arg(long)]
    cache_ttl: Option<u64>,

    /// URL of the job posting
    url: String,

    /// Output path for the markdown file (derived from content if omitted)
    output_path: Option<String>,
}

/// Remove characters that are problematic in file/directory names.
fn sanitize_for_path(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '-' || *c == '_')
        .collect::<String>()
        .trim()
        .to_string()
}

/// Derive an output path from the extracted markdown content.
///
/// Looks for a `# Title` heading and a company name in various formats:
/// `**Company:** Name`, `## Company\nName`, etc.
/// Falls back to "job" / "unknown" when not found.
fn derive_output_path(markdown: &str) -> PathBuf {
    let mut title = None;
    let mut company = None;
    let lines: Vec<&str> = markdown.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // First h1 is the job title
        if title.is_none() {
            if let Some(rest) = trimmed.strip_prefix("# ") {
                if !rest.trim().is_empty() {
                    title = Some(rest.trim().to_string());
                }
            }
        }

        // Company in bold-label format: **Company:** Name
        if company.is_none() {
            if let Some(rest) = trimmed.strip_prefix("**Company:**") {
                let val = rest.trim();
                if !val.is_empty() {
                    company = Some(val.to_string());
                    continue;
                }
            }
        }

        // Company as h2 section: ## Company\n<next non-empty line>
        if company.is_none() && trimmed == "## Company" {
            if let Some(next) = lines.get(i + 1) {
                let next = next.trim();
                if !next.is_empty() && !next.starts_with('#') {
                    company = Some(next.to_string());
                }
            }
        }

        if title.is_some() && company.is_some() {
            break;
        }
    }

    let company_dir = sanitize_for_path(&company.unwrap_or_else(|| "unknown".to_string()));
    let file_stem = sanitize_for_path(&title.unwrap_or_else(|| "job".to_string()));

    let company_dir = if company_dir.is_empty() { "unknown".to_string() } else { company_dir };
    let file_stem = if file_stem.is_empty() { "job".to_string() } else { file_stem };

    PathBuf::from(format!("jobs/{}/{}.md", company_dir, file_stem))
}

/// Try to extract job content from ld+json and meta tags when the page body
/// is empty (common with JS-rendered SPAs like Workday).
fn extract_from_metadata(html: &str) -> Option<String> {
    // Extract ld+json block
    let ld_json = html
        .find(r#"<script type="application/ld+json">"#)
        .and_then(|start| {
            let content_start = start + r#"<script type="application/ld+json">"#.len();
            html[content_start..]
                .find("</script>")
                .map(|end| html[content_start..content_start + end].trim())
        });

    if let Some(json_str) = ld_json {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
            let title = val["title"].as_str().or(val["identifier"]["name"].as_str());
            let company = val["hiringOrganization"]["name"].as_str();
            let location = val["jobLocation"]["address"]["addressLocality"].as_str();
            let country = val["jobLocation"]["address"]["addressCountry"].as_str();
            let description = val["description"].as_str();
            let date = val["datePosted"].as_str();

            if title.is_some() || description.is_some() {
                let mut parts = Vec::new();

                if let Some(t) = title {
                    parts.push(format!("Job Title: {}", t));
                }
                if let Some(c) = company {
                    parts.push(format!("Company: {}", c));
                }
                if let Some(loc) = location {
                    if let Some(ctry) = country {
                        parts.push(format!("Location: {}, {}", loc, ctry));
                    } else {
                        parts.push(format!("Location: {}", loc));
                    }
                }
                if let Some(d) = date {
                    parts.push(format!("Date Posted: {}", d));
                }
                if let Some(desc) = description {
                    parts.push(format!("\nJob Description:\n{}", desc));
                }

                return Some(parts.join("\n"));
            }
        }
    }

    // Fallback: og:description meta tag
    let og_desc = html
        .find(r#"property="og:description" content=""#)
        .or_else(|| html.find(r#"name="description" property="og:description" content=""#))
        .and_then(|pos| {
            let attr = r#"content=""#;
            let content_start = html[pos..].find(attr).map(|i| pos + i + attr.len())?;
            html[content_start..].find('"').map(|end| &html[content_start..content_start + end])
        });

    let og_title = html
        .find(r#"property="og:title" content=""#)
        .and_then(|pos| {
            let attr = r#"content=""#;
            let content_start = html[pos..].find(attr).map(|i| pos + i + attr.len())?;
            html[content_start..].find('"').map(|end| &html[content_start..content_start + end])
        });

    match (og_title, og_desc) {
        (Some(title), Some(desc)) => Some(format!("Job Title: {}\n\n{}", title, desc)),
        (None, Some(desc)) => Some(desc.to_string()),
        _ => None,
    }
}

async fn fetch_and_extract<M: CompletionModel + Clone>(
    url: &str,
    output_path: Option<&str>,
    model_name: &str,
    completion_model: &M,
    cache: Option<&CacheConfig>,
) -> Result<()> {
    eprintln!("Fetching {}...", url);
    let html = reqwest::get(url)
        .await
        .with_context(|| format!("failed to fetch URL: {}", url))?
        .text()
        .await
        .with_context(|| format!("failed to read response body from: {}", url))?;

    eprintln!("Converting HTML to markdown...");
    let converter = htmd::HtmlToMarkdown::builder()
        .skip_tags(SKIP_TAGS.to_vec())
        .build();
    let raw_md = converter
        .convert(&html)
        .with_context(|| "failed to convert HTML to markdown")?;

    // If htmd produced empty content (JS-rendered SPA), fall back to metadata
    let content = if raw_md.trim().is_empty() {
        eprintln!("Page body is empty (JS-rendered). Extracting from metadata...");
        extract_from_metadata(&html)
            .ok_or_else(|| anyhow!("page has no visible content and no extractable metadata"))?
    } else {
        raw_md
    };

    eprintln!("Extracting job posting with {}...", model_name);
    let result = prompt_text_with_temperature(
        completion_model,
        pipelines::prompts::JOB_EXTRACT_PREAMBLE,
        &content,
        0.2,
        cache,
    )
    .await?;

    let out_path = match output_path {
        Some(p) => PathBuf::from(p),
        None => derive_output_path(&result.value),
    };

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create output dir: {}", parent.display()))?;
        }
    }
    fs::write(&out_path, &result.value)
        .with_context(|| format!("failed to write output to {}", out_path.display()))?;

    eprintln!("Saved job posting to {}", out_path.display());
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    pipelines::llm::enable_spinners(console::Term::stderr().is_term());

    let model_name = args
        .model
        .clone()
        .unwrap_or_else(|| args.provider.default_model().to_string());

    let cache = CacheConfig {
        provider_name: args.provider.name().to_string(),
        model_name: model_name.clone(),
        enabled: !args.no_cache,
        max_age: args.cache_ttl.map(std::time::Duration::from_secs),
    };
    let cache_opt = Some(&cache);

    pipelines::dispatch_completion!(args.provider, &model_name, |m| {
        fetch_and_extract(&args.url, args.output_path.as_deref(), &model_name, &m, cache_opt).await
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- derive_output_path tests ---

    #[test]
    fn derive_output_path_title_and_bold_company() {
        let md = "# Senior Engineer\n\n**Company:** Acme Corp\n\nSome description";
        let path = derive_output_path(md);
        assert_eq!(path, PathBuf::from("jobs/Acme Corp/Senior Engineer.md"));
    }

    #[test]
    fn derive_output_path_title_and_h2_company() {
        let md = "# Backend Developer\n\n## Company\nWidgetCo\n\nDetails here";
        let path = derive_output_path(md);
        assert_eq!(path, PathBuf::from("jobs/WidgetCo/Backend Developer.md"));
    }

    #[test]
    fn derive_output_path_missing_title_falls_back() {
        let md = "No heading here\n\n**Company:** Acme";
        let path = derive_output_path(md);
        assert_eq!(path, PathBuf::from("jobs/Acme/job.md"));
    }

    #[test]
    fn derive_output_path_missing_company_falls_back() {
        let md = "# Software Engineer\n\nNo company info";
        let path = derive_output_path(md);
        assert_eq!(path, PathBuf::from("jobs/unknown/Software Engineer.md"));
    }

    #[test]
    fn derive_output_path_sanitizes_special_chars() {
        let md = "# Staff Engineer (Remote)\n\n**Company:** Acme/Corp™";
        let path = derive_output_path(md);
        let file_stem = path.file_stem().unwrap().to_str().unwrap();
        let company_dir = path.parent().unwrap().file_name().unwrap().to_str().unwrap();
        // Special chars like (, ), /, ™ should be stripped from components
        assert!(!file_stem.contains('('));
        assert!(!file_stem.contains(')'));
        assert!(!company_dir.contains('/'));
        assert!(!company_dir.contains('™'));
    }

    // --- extract_from_metadata tests ---

    #[test]
    fn extract_from_metadata_ld_json_all_fields() {
        let html = r#"<html><head><script type="application/ld+json">{
            "title": "Software Engineer",
            "hiringOrganization": {"name": "Acme"},
            "jobLocation": {"address": {"addressLocality": "NYC", "addressCountry": "US"}},
            "datePosted": "2025-01-01",
            "description": "Build amazing things"
        }</script></head></html>"#;
        let result = extract_from_metadata(html).unwrap();
        assert!(result.contains("Software Engineer"));
        assert!(result.contains("Acme"));
        assert!(result.contains("NYC"));
        assert!(result.contains("US"));
        assert!(result.contains("Build amazing things"));
    }

    #[test]
    fn extract_from_metadata_ld_json_missing_fields() {
        let html = r#"<html><head><script type="application/ld+json">{
            "title": "Engineer",
            "description": "Do stuff"
        }</script></head></html>"#;
        let result = extract_from_metadata(html).unwrap();
        assert!(result.contains("Engineer"));
        assert!(result.contains("Do stuff"));
        assert!(!result.contains("Company"));
    }

    #[test]
    fn extract_from_metadata_og_description_fallback() {
        let html = r#"<html><head>
            <meta property="og:title" content="Great Job">
            <meta property="og:description" content="Join our team">
        </head></html>"#;
        let result = extract_from_metadata(html).unwrap();
        assert!(result.contains("Great Job"));
        assert!(result.contains("Join our team"));
    }

    #[test]
    fn extract_from_metadata_no_metadata_returns_none() {
        let html = "<html><body>Hello</body></html>";
        assert!(extract_from_metadata(html).is_none());
    }

    // --- sanitize_for_path tests ---

    #[test]
    fn sanitize_for_path_keeps_alphanumeric_and_safe_chars() {
        assert_eq!(sanitize_for_path("Hello World-Test_123"), "Hello World-Test_123");
    }

    #[test]
    fn sanitize_for_path_strips_unsafe_chars() {
        assert_eq!(sanitize_for_path("file/name:with*bad"), "filenamewithbad");
    }
}
