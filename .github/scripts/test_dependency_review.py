"""Checks for incomplete snapshots and pip-report inventory conversion."""

import base64
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import dependency_review as review


def pip_report():
    return {"version": "1", "install": [
        {"metadata": {"name": "LangChain_Core", "version": "1.7.4"}, "requested": True},
        {"metadata": {"name": "typing-extensions", "version": "4.15.0"}, "requested": False},
    ]}


class DependencyReviewTests(unittest.TestCase):
    def setUp(self):
        output_capture = redirect_stdout(io.StringIO())
        output_capture.__enter__()
        self.addCleanup(output_capture.__exit__, None, None, None)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        environment = patch.dict(os.environ, {
            "GITHUB_REPOSITORY": "owner/repo", "BASE_SHA": "a" * 40,
            "HEAD_SHA": "b" * 40, "SNAPSHOT_SHA": "b" * 40,
            "SNAPSHOT_REF": "refs/heads/fix", "GITHUB_RUN_ID": "123",
            "GITHUB_TOKEN": "test-token", "GITHUB_API_URL": "https://api.github.com",
        })
        environment.start()
        self.addCleanup(environment.stop)

    def test_transitive_versions_are_retained_under_original_manifest(self):
        manifest = review.manifest_from_report(".github/requirements/pylint.in", pip_report())
        self.assertEqual(manifest["file"]["source_location"], ".github/requirements/pylint.in")
        direct = manifest["resolved"]["langchain-core"]
        self.assertEqual(direct["package_url"], "pkg:pypi/langchain-core@1.7.4")
        self.assertEqual(direct["relationship"], "direct")
        self.assertEqual(direct["scope"], "development")
        self.assertEqual(manifest["resolved"]["typing-extensions"]["relationship"], "indirect")

    def test_empty_or_unknown_report_cannot_be_submitted(self):
        for report in ({}, {"version": "1", "install": []}, {**pip_report(), "version": "2"}):
            with self.subTest(report=report), self.assertRaises(ValueError):
                review.manifest_from_report("requirements.txt", report)

    def test_duplicate_normalized_packages_are_rejected(self):
        report = pip_report()
        report["install"].append({"metadata": {"name": "langchain-core", "version": "0.3.86"}, "requested": True})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            review.manifest_from_report("requirements.txt", report)

    def test_direct_url_is_not_misreported_as_a_pypi_release(self):
        report = pip_report()
        report["install"][0]["is_direct"] = True
        with self.assertRaises(ValueError):
            review.manifest_from_report("requirements.txt", report)

    def test_source_distribution_is_included_in_inventory(self):
        report = {"version": "1", "install": [{
            "metadata": {"name": "pydotplus", "version": "2.0.2"},
            "requested": False, "is_direct": False,
            "download_info": {"url": "https://files.pythonhosted.org/packages/pydotplus-2.0.2.tar.gz"},
        }]}
        manifest = review.manifest_from_report("requirements.txt", report)
        self.assertEqual(manifest["resolved"]["pydotplus"]["package_url"], "pkg:pypi/pydotplus@2.0.2")
        self.assertEqual(manifest["resolved"]["pydotplus"]["relationship"], "indirect")

    def test_incomplete_comparison_fails_instead_of_passing(self):
        warning = base64.b64encode(b"No snapshots were found for the head SHA").decode()
        with patch.object(review, "github_request", return_value=([], {
            "x-github-dependency-graph-snapshot-warnings": warning,
        })), self.assertRaisesRegex(RuntimeError, "incomplete"):
            review.require_complete_comparison(self.output, timeout=0)

    def test_hundred_results_without_next_link_do_not_loop(self):
        with patch.object(review, "github_request", return_value=([{}] * 100, {})) as api:
            review.require_complete_comparison(self.output, timeout=0)
        self.assertEqual(api.call_count, 1)

    def test_warnings_on_later_pages_are_not_lost(self):
        next_url = "https://api.github.com/repos/owner/repo/dependency-graph/compare/" + "a" * 40 + "..." + "b" * 40 + "?page=2"
        warning = base64.b64encode(b"Missing base snapshot").decode()
        responses = [([{}], {"Link": f'<{next_url}>; rel="next"'}),
                     ([], {"x-github-dependency-graph-snapshot-warnings": warning})]
        with patch.object(review, "github_request", side_effect=responses) as api:
            with self.assertRaisesRegex(RuntimeError, "Missing base"):
                review.require_complete_comparison(self.output, timeout=0)
        self.assertEqual(api.call_count, 2)

    def test_foreign_pagination_url_is_rejected(self):
        with patch.object(review, "github_request", return_value=([], {
            "Link": '<https://example.invalid/next>; rel="next"',
        })), self.assertRaisesRegex(RuntimeError, "pagination URL"):
            review.require_complete_comparison(self.output, timeout=0)

    def test_wrong_checkout_is_rejected_before_resolution_or_submission(self):
        with patch.object(review.subprocess, "check_output", return_value="a" * 40), \
                patch.object(review.subprocess, "run") as pip, \
                patch.object(review, "github_request") as api:
            with self.assertRaisesRegex(ValueError, "does not match"):
                review.submit_snapshot(self.output, self.output)
        pip.assert_not_called()
        api.assert_not_called()

    def test_failed_resolution_never_submits_partial_inventory(self):
        source = self.output / "source"
        source.mkdir()
        (source / "requirements.txt").write_text("langchain==1.4.8", encoding="utf-8")
        with patch.object(review.subprocess, "check_output", side_effect=["b" * 40, b""]), \
                patch.object(review.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "pip")), \
                patch.object(review, "github_request") as api:
            with self.assertRaises(subprocess.CalledProcessError):
                review.submit_snapshot(source, self.output / "reports")
        api.assert_not_called()

    def test_all_manifests_are_resolved_before_one_submission(self):
        source = self.output / "source"
        for manifest in review.MANIFESTS:
            path = source / manifest
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("langchain==1.4.8", encoding="utf-8")

        def resolve(command, **kwargs):
            self.assertNotIn("GITHUB_TOKEN", kwargs["env"])
            self.assertIn("--ignore-installed", command)
            self.assertIn("--dry-run", command)
            report_path = Path(command[command.index("--report") + 1])
            report_path.write_text(json.dumps(pip_report()), encoding="utf-8")

        for result in ("SUCCESS", "ACCEPTED", "INVALID", "UNKNOWN"):
            with self.subTest(result=result), \
                    patch.object(review.subprocess, "check_output", side_effect=["b" * 40, b""]), \
                    patch.object(review.subprocess, "run", side_effect=resolve) as pip, \
                    patch.object(review, "github_request", return_value=({"result": result, "id": 1}, {})) as api:
                if result in {"SUCCESS", "ACCEPTED"}:
                    review.submit_snapshot(source, self.output / "reports")
                else:
                    with self.assertRaisesRegex(RuntimeError, "did not accept"):
                        review.submit_snapshot(source, self.output / "reports")
            self.assertEqual(pip.call_count, 4)
            snapshot = api.call_args.args[1]
            self.assertEqual(set(snapshot["manifests"]), set(review.MANIFESTS))
            self.assertEqual(snapshot["sha"], "b" * 40)
            self.assertTrue(snapshot["job"]["id"].endswith("b" * 40))
            api.assert_called_once()


if __name__ == "__main__":
    unittest.main()
