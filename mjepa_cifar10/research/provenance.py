from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from .models import StudySpec


PROTECTED_BRANCHES: Final = frozenset(("main", "master"))
STUDY_BRANCH_PREFIX: Final[str] = "codex/research/"


@dataclass(frozen=True)
class GitProvenance:
    path: str
    sha: str
    branch: str
    dirty: bool
    upstream: str | None
    pushed: bool


@dataclass(frozen=True)
class ProvenanceReport:
    parent: GitProvenance
    mjepa: GitProvenance
    vit: GitProvenance
    lockfile_sha256: str
    installed_sources: dict[str, dict[str, Any]]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_provenance(repo: Path) -> GitProvenance:
    sha = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "branch", "--show-current")
    dirty = bool(_git(repo, "status", "--porcelain"))
    upstream_result = subprocess.run(
        ("git", "-C", str(repo), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"),
        check=False,
        capture_output=True,
        text=True,
    )
    upstream = upstream_result.stdout.strip() if upstream_result.returncode == 0 else None
    pushed = upstream is not None and _git(repo, "rev-parse", "@{u}") == sha
    return GitProvenance(str(repo.resolve()), sha, branch, dirty, upstream, pushed)


def _installed_source(distribution_name: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(distribution_name)
    raw = distribution.read_text("direct_url.json")
    return json.loads(raw) if raw else {"version": distribution.version}


def collect_provenance(spec: StudySpec, repo_root: Path) -> ProvenanceReport:
    parent = git_provenance(repo_root)
    mjepa = git_provenance((repo_root / ".." / "mjepa").resolve())
    vit = git_provenance((repo_root / ".." / "vit").resolve())
    lockfile_path = repo_root / "uv.lock"
    lockfile_sha256 = hashlib.sha256(lockfile_path.read_bytes()).hexdigest()
    installed_sources = {name: _installed_source(name) for name in ("mjepa", "vit")}
    errors: list[str] = []
    if parent.dirty:
        errors.append("parent repository is dirty")
    if parent.branch in PROTECTED_BRANCHES or not parent.branch.startswith(STUDY_BRANCH_PREFIX):
        errors.append(f"parent branch must start with {STUDY_BRANCH_PREFIX!r}")
    if not parent.pushed:
        errors.append("parent branch is not pushed at its current SHA")
    for name, provenance in (("mjepa", mjepa), ("vit", vit)):
        if provenance.dirty:
            errors.append(f"{name} repository is dirty")
        if provenance.branch in PROTECTED_BRANCHES or not provenance.branch.startswith(STUDY_BRANCH_PREFIX):
            errors.append(f"{name} branch must start with {STUDY_BRANCH_PREFIX!r}")
    expected_shas = {"parent": parent, "mjepa": mjepa, "vit": vit}
    for name, expected_sha in spec.code_shas.items():
        if name in expected_shas and expected_sha not in ("", "REQUIRED"):
            actual_sha = expected_shas[name].sha
            if actual_sha != expected_sha:
                errors.append(f"{name} SHA mismatch: expected {expected_sha}, got {actual_sha}")
    for name in ("mjepa", "vit"):
        expected_sha = spec.code_shas.get(name)
        source = installed_sources[name]
        installed_sha = source.get("vcs_info", {}).get("commit_id")
        if expected_sha and expected_sha != "REQUIRED" and installed_sha != expected_sha:
            errors.append(f"installed {name} source does not match recorded SHA {expected_sha}")
        if source.get("dir_info", {}).get("editable"):
            errors.append(f"installed {name} is editable; build the frozen study environment before launch")
    return ProvenanceReport(parent, mjepa, vit, lockfile_sha256, installed_sources, tuple(errors))


def assert_launch_provenance(spec: StudySpec, repo_root: Path) -> ProvenanceReport:
    result = subprocess.run(
        ("uv", "lock", "--check"),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"uv.lock is stale: {result.stderr.strip() or result.stdout.strip()}")
    report = collect_provenance(spec, repo_root)
    if report.errors:
        raise RuntimeError("launch provenance rejected:\n- " + "\n- ".join(report.errors))
    return report
