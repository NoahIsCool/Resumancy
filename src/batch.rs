use std::fs;
use std::path::{Path, PathBuf};
use anyhow::Context;

/// Recursively collect all `.md` files under `dir`.
pub fn find_md_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut results = Vec::new();
    find_md_files_recursive(dir, &mut results)?;
    results.sort();
    Ok(results)
}

fn find_md_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir)
        .with_context(|| format!("failed to read directory: {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            find_md_files_recursive(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "md") {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn empty_directory_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let result = find_md_files(dir.path()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn single_md_file_found() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("job.md"), "content").unwrap();
        let result = find_md_files(dir.path()).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].ends_with("job.md"));
    }

    #[test]
    fn recursive_discovery() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        fs::create_dir(&sub).unwrap();
        fs::write(dir.path().join("a.md"), "").unwrap();
        fs::write(sub.join("b.md"), "").unwrap();
        let result = find_md_files(dir.path()).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn non_md_files_ignored() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("readme.txt"), "").unwrap();
        fs::write(dir.path().join("data.json"), "").unwrap();
        fs::write(dir.path().join("job.md"), "").unwrap();
        let result = find_md_files(dir.path()).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn results_are_sorted() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("c.md"), "").unwrap();
        fs::write(dir.path().join("a.md"), "").unwrap();
        fs::write(dir.path().join("b.md"), "").unwrap();
        let result = find_md_files(dir.path()).unwrap();
        let names: Vec<_> = result.iter().map(|p| p.file_name().unwrap().to_str().unwrap()).collect();
        assert_eq!(names, vec!["a.md", "b.md", "c.md"]);
    }
}
