#!/usr/bin/env python3

"""
FraudShield AI
PHASE 01: Repository Inspection & Project Setup

Responsibilities:
1. Verify reference repositories.
2. Inspect licences.
3. Inspect dependencies.
4. Detect reusable components.
5. Create project structure.
6. Generate inspection report.
"""

from pathlib import Path
from datetime import datetime
import json
import sys


ROOT = Path(__file__).resolve().parent.parent

REFERENCES = {
    "Meshwatch": ROOT / "reference-meshwatch",
    "Driftline": ROOT / "reference-driftline",
}

PROJECT_DIRS = [
    "data/raw",
    "data/processed",
    "data/synthetic",
    "models",
    "reports",
    "screenshots",
    "AIML-01",
    "deployment/docker",
    "deployment/kubernetes",
    "feature_store",
    "streaming",
    "backend",
    "dashboard",
    "monitoring",
    "retraining",
    "tests",
]

REQUIREMENTS = {
    "dataset": ["dataset", "data"],
    "eda": ["eda", "notebooks"],
    "feature_engineering": ["features", "feature"],
    "feast": ["feast", "feature_store"],
    "xgboost": ["xgboost", "model"],
    "mlflow": ["mlflow", "experiment"],
    "graph_neural_network": ["gnn", "graphsage", "graph"],
    "kafka": ["kafka", "streaming"],
    "fastapi": ["fastapi", "serving"],
    "evidently": ["evidently", "drift"],
    "retraining": ["retrain"],
    "dashboard": ["dashboard"],
    "kubernetes": ["k8s", "kubernetes"],
}


def inspect_repository(name, path):

    result = {
        "name": name,
        "exists": path.exists(),
        "files": [],
        "license": None,
        "dependencies": [],
        "detected_capabilities": [],
    }

    if not path.exists():
        return result

    files = [
        p for p in path.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
    ]

    relative_files = [
        str(p.relative_to(path))
        for p in files
    ]

    result["files"] = relative_files

    # Licence inspection
    licenses = list(path.glob("LICENSE*"))

    if licenses:
        result["license"] = licenses[0].read_text(
            encoding="utf-8",
            errors="ignore"
        )[:5000]

    # Dependency inspection
    dependency_files = [
        "requirements.txt",
        "pyproject.toml",
    ]

    for filename in dependency_files:

        dependency_path = path / filename

        if dependency_path.exists():

            result["dependencies"].append({
                "file": filename,
                "content": dependency_path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )[:10000],
            })

    # Capability detection
    searchable = " ".join(relative_files).lower()

    for capability, keywords in REQUIREMENTS.items():

        if any(keyword in searchable for keyword in keywords):

            result["detected_capabilities"].append(
                capability
            )

    return result


def create_project_structure():

    print("\nCreating project structure...")

    for directory in PROJECT_DIRS:

        path = ROOT / directory
        path.mkdir(parents=True, exist_ok=True)

        print(f"[CREATED/VERIFIED] {directory}")


def inspect_environment():

    print("\nChecking Python environment...")

    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "project_root": str(ROOT),
        "venv_active": (
            sys.prefix != sys.base_prefix
        ),
    }


def main():

    print("=" * 65)
    print("       FRAUDSHIELD AI - PHASE 01")
    print("       REPOSITORY INSPECTION")
    print("=" * 65)

    create_project_structure()

    repository_results = {}

    for name, path in REFERENCES.items():

        print(f"\nInspecting {name}...")

        result = inspect_repository(name, path)

        repository_results[name] = result

        print("Repository exists:", result["exists"])

        print("Files found:", len(result["files"]))

        print(
            "Detected capabilities:",
            ", ".join(result["detected_capabilities"])
        )

        if result["license"]:
            print("LICENSE FOUND")
        else:
            print("WARNING: LICENSE NOT FOUND")

    environment = inspect_environment()

    report = {
        "project": "FraudShield AI",
        "phase": 1,
        "generated_at": datetime.now().isoformat(),
        "environment": environment,
        "repositories": repository_results,
        "status": "INSPECTION_COMPLETED",
    }

    report_path = ROOT / "reports" / "phase01_report.json"

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("PHASE 01 REPORT")
    print("=" * 65)

    print("Python:", environment["python_version"].split()[0])

    print("Virtual environment:", environment["venv_active"])

    print("Report:", report_path)

    print("\nPHASE 01 INSPECTION COMPLETED")

    if not all(
        result["exists"]
        for result in repository_results.values()
    ):
        raise SystemExit(
            "ERROR: One or more reference repositories are missing."
        )

    if not environment["venv_active"]:
        print(
            "WARNING: Activate .venv before installing dependencies."
        )


if __name__ == "__main__":
    main()
