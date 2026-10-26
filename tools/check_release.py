"""Check source integrity, import closure, local links and accidental private files."""

import ast
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    provenance = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    for item in provenance["sources"]:
        path = ROOT / item["destination"]
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        actual = hashlib.sha256(text.encode()).hexdigest()
        if actual != item["packaged_normalized_sha256"]:
            errors.append(f"Changed packaged source hash: {item['destination']}")
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".pem", ".key", ".zip", ".xls", ".xlsx", ".pdf"}:
            errors.append(f"Unexpected non-source/private artifact: {rel}")
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"Large source artifact: {rel}")
        if path.suffix.lower() not in {".py", ".md", ".json", ".txt", ".yml", ".yaml", ".cff"}:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in (r"/mnt/" + "cr_", r"C:\\Users\\", r"D:\\PythonCode\\", r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY", r"gh[pousr]_[A-Za-z0-9]{30,}"):
            if re.search(pattern, text):
                errors.append(f"Possible private path or credential in {rel}")
        if path.suffix == ".py":
            try:
                tree = ast.parse(text, filename=rel)
            except SyntaxError as error:
                errors.append(str(error))
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("ours"):
                        errors.append(f"Unconverted original-project import in {rel}")
                    if node.module.startswith("cpr_sam3"):
                        target = ROOT.joinpath(*node.module.split("."))
                        if not target.with_suffix(".py").is_file() and not (target / "__init__.py").is_file():
                            errors.append(f"Missing local module {node.module} in {rel}")
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^\s)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                if not (path.parent / target.split("#")[0]).exists():
                    errors.append(f"Broken local Markdown link {target} in {rel}")
    for name in ("vaihingen", "potsdam", "loveda", "openearthmap"):
        profile = json.loads((ROOT / "configs" / f"{name}.json").read_text())
        bank = json.loads((ROOT / "configs/prompt_banks" / profile["prompt_bank"]).read_text())
        if not bank["classes"] or any(not item.get("prompts") for item in bank["classes"]):
            errors.append(f"Empty prompt bank: {name}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Release checks: {'FAIL' if errors else 'PASS'} ({len(provenance['sources'])} tracked source units)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
