import json
import re
from pathlib import Path

CASES_DIR = Path(__file__).parent.parent / "cases"
ID_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
KNOWN_CATEGORIES = {"code_review", "doc_summary", "fact_answer"}
REQUIRED_KEYS = (
    "id",
    "category",
    "question",
    "context_old",
    "context_new",
    "expect_new",
)


def _load_all_cases():
    cases = []
    for path in sorted(CASES_DIR.glob("*.json")):
        for case in json.loads(path.read_text()):
            cases.append((path.name, case))
    return cases


def test_case_schema():
    cases = _load_all_cases()
    assert cases, "no case files found"

    violations = []
    seen_ids: dict[str, list[str]] = {}
    for filename, case in cases:
        label = f"{filename}:{case.get('id', '<missing id>')}"

        missing = [key for key in REQUIRED_KEYS if key not in case]
        if missing:
            violations.append(f"{label} missing keys {missing}")
            continue

        if not ID_PATTERN.match(case["id"]):
            violations.append(f"{label} id is not kebab-case")
        seen_ids.setdefault(case["id"], []).append(filename)

        if case["category"] not in KNOWN_CATEGORIES:
            violations.append(f"{label} unknown category {case['category']!r}")

        if not case["question"].strip():
            violations.append(f"{label} has an empty question")

        if case["context_old"] == case["context_new"]:
            violations.append(f"{label} context_old equals context_new (no edit)")

        if not isinstance(case["expect_new"], list) or not all(
            isinstance(s, str) for s in case["expect_new"]
        ):
            violations.append(f"{label} expect_new must be a list of strings")

        if "forbid_new" in case and (
            not isinstance(case["forbid_new"], list)
            or not all(isinstance(s, str) for s in case["forbid_new"])
        ):
            violations.append(f"{label} forbid_new must be a list of strings")

    for case_id, files in seen_ids.items():
        if len(files) > 1:
            violations.append(f"id {case_id!r} duplicated across {files}")

    assert not violations, "\n".join(violations)
