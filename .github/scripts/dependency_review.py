"""Submit explicit pip manifests and require a complete dependency comparison."""

import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


MANIFESTS = (
    "requirements.txt",
    ".github/requirements/pylint.in",
    ".github/requirements/pylint-py311-linux.txt",
    ".github/requirements/release-build-py311-linux.txt",
)
DETECTOR = {
    "name": "hana-ai-pip-report",
    "version": "1.0.0",
    "url": "https://github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud",
}


def github_request(path, payload=None):
    """Call the official API without exposing the token in command arguments."""
    endpoint = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "hana-ai-dependency-review",
    }
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = Request(endpoint + path, data=data, headers=headers)
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response), response.headers
    except HTTPError as exc:
        raise RuntimeError(f"GitHub dependency API returned HTTP {exc.code}") from None


def manifest_from_report(path, report):
    """Retain every resolved version, including transitive dependencies."""
    if str(report.get("version")) != "1" or not report.get("install"):
        raise ValueError(f"Missing or unsupported pip report for {path}")
    resolved = {}
    for item in report["install"]:
        metadata = item["metadata"]
        name = re.sub(r"[-_.]+", "-", metadata["name"]).lower()
        version = metadata["version"]
        if not name or not version or item.get("is_direct"):
            raise ValueError(f"Cannot represent an unnamed or direct-URL dependency in {path}")
        if name in resolved:
            raise ValueError(f"Duplicate package in pip report: {name}")
        resolved[name] = {
            "package_url": f"pkg:pypi/{quote(name, safe='-._~')}@{quote(version, safe='-._~')}",
            "relationship": "direct" if item["requested"] else "indirect",
            "scope": "development" if path.startswith(".github/") else "runtime",
        }
    return {"name": path, "file": {"source_location": path}, "resolved": resolved}


def submit_snapshot(source, output):
    """Resolve the exact checked-out tree before submitting a single snapshot."""
    source = source.resolve()
    sha = os.environ["SNAPSHOT_SHA"]
    actual_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if actual_sha != sha:
        raise ValueError("The source checkout does not match SNAPSHOT_SHA")
    if subprocess.check_output(["git", "diff", "HEAD", "--", *MANIFESTS], cwd=source):
        raise ValueError("Dependency manifests differ from the checked-out commit")
    output.mkdir(parents=True, exist_ok=True)
    manifests = {}
    pip_environment = dict(os.environ)
    pip_environment.pop("GITHUB_TOKEN", None)
    pip_environment.pop("PYTHONPATH", None)
    for index, path in enumerate(MANIFESTS):
        if not (source / path).is_file():
            raise FileNotFoundError(f"Expected dependency manifest is missing: {path}")
        report_path = (output / f"pip-report-{index}.json").resolve()
        # --ignore-installed prevents packages on the runner from disappearing
        # from the report. Binary-only resolution avoids running package builds.
        command = [
            sys.executable, "-m", "pip", "install",
            "--dry-run", "--ignore-installed", "--only-binary=:all:",
            "--disable-pip-version-check", "--report", str(report_path),
            "-r", path,
        ]
        if path.endswith("-py311-linux.txt"):
            command.append("--require-hashes")
        subprocess.run(command, cwd=source, env=pip_environment, check=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifests[path] = manifest_from_report(path, report)
        print(f"Resolved {path}: {len(manifests[path]['resolved'])} packages", flush=True)
    repository = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["GITHUB_RUN_ID"]
    snapshot = {
        "version": 0,
        "sha": sha,
        "ref": os.environ["SNAPSHOT_REF"],
        "job": {
            "id": run_id,
            # Keep this identical for base/head and across subsequent runs.
            "correlator": "hana-ai-python311-linux-dependencies",
            "html_url": f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{repository}/actions/runs/{run_id}",
        },
        "detector": DETECTOR,
        "scanned": datetime.now(timezone.utc).isoformat(),
        "manifests": manifests,
    }
    (output / "snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    response, _ = github_request(f"/repos/{repository}/dependency-graph/snapshots", snapshot)
    if response.get("result") != "SUCCESS":
        raise RuntimeError("GitHub did not accept the dependency snapshot")
    print(f"GitHub accepted snapshot {response.get('id')} for {sha}", flush=True)


def require_complete_comparison(output, timeout=180):
    """Missing snapshots must fail, rather than merely warn and continue."""
    repository = os.environ["GITHUB_REPOSITORY"]
    base, head = os.environ["BASE_SHA"], os.environ["HEAD_SHA"]
    deadline = time.monotonic() + timeout
    while True:
        changes = []
        warnings = set()
        comparison_path = f"/repos/{repository}/dependency-graph/compare/{base}...{head}"
        request_path = comparison_path + "?per_page=100"
        visited = set()
        while request_path:
            if request_path in visited:
                raise RuntimeError("GitHub returned a repeated pagination link")
            visited.add(request_path)
            result, headers = github_request(request_path)
            if not isinstance(result, list):
                raise ValueError("Unexpected GitHub dependency comparison response")
            changes.extend(result)
            encoded = headers.get("x-github-dependency-graph-snapshot-warnings", "")
            if encoded:
                warning = base64.b64decode(encoded).decode("utf-8").strip()
                if warning:
                    warnings.add(warning)
            next_page = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", ""))
            request_path = None
            if next_page:
                endpoint = os.environ.get("GITHUB_API_URL", "https://api.github.com")
                if not next_page.group(1).startswith(endpoint + comparison_path + "?"):
                    raise RuntimeError("Unexpected dependency comparison pagination URL")
                request_path = next_page.group(1)[len(endpoint):]
        output.mkdir(parents=True, exist_ok=True)
        (output / "comparison.json").write_text(json.dumps(changes, indent=2), encoding="utf-8")
        (output / "snapshot-warnings.json").write_text(json.dumps(sorted(warnings)), encoding="utf-8")
        if not warnings:
            print(f"GitHub comparison ready: {len(changes)} dependency changes, no snapshot warnings")
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("GitHub dependency comparison is incomplete: " + "; ".join(sorted(warnings)))
        print("Waiting for GitHub to index dependency snapshots", flush=True)
        time.sleep(min(10, max(0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("submit", "check"))
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "submit":
        submit_snapshot(args.source, args.output)
    else:
        require_complete_comparison(args.output)


if __name__ == "__main__":
    main()
