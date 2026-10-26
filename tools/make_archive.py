"""Build a source-only ZIP outside this repository, excluding runtime artifacts."""

import argparse
import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", "__pycache__", ".pytest_cache", ".venv", "data", "checkpoints", "third_party", "artifacts", "work_dirs"}
EXTENSIONS = {".py", ".md", ".json", ".txt", ".yml", ".yaml", ".cff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.is_relative_to(ROOT):
        parser.error("Write the archive outside the repository to avoid recursive inclusion")
    files = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if not path.is_file() or set(rel.parts) & EXCLUDED or path.name == "paths.json":
            continue
        if path.suffix in EXTENSIONS or path.name == ".gitignore":
            files.append(path)
    if output.exists():
        parser.error("Output already exists; use a new archive name")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, (Path("CPR-SAM3") / path.relative_to(ROOT)).as_posix())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{sha}  {output.name}\n", encoding="ascii")
    print(f"Source archive: {output.name}; files: {len(files)}; bytes: {output.stat().st_size}; SHA256: {sha}")


if __name__ == "__main__":
    main()
