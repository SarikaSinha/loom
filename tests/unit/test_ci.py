import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[2] / ".github/scripts/ci.py"
    spec = importlib.util.spec_from_file_location("loom_ci", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_mend_inputs_fail_before_network(gate, monkeypatch):
    for name in ("MEND_CLI_VERSION", "MEND_CLI_SHA256", "MEND_SCOPE"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(gate.GateFailure, match="missing_approved_mend_inputs"):
        gate.mend()


def test_sast_requires_evidence_and_rejects_findings(gate, tmp_path):
    with pytest.raises(gate.GateFailure, match="missing_sast_evidence"):
        gate.check_sarif(tmp_path)
    path = tmp_path / "python.sarif"
    report = {"version": "2.1.0", "runs": [{"results": []}]}
    path.write_text(json.dumps(report))
    gate.check_sarif(tmp_path)
    report["runs"][0]["results"] = [{"level": "warning", "ruleId": "test"}]
    path.write_text(json.dumps(report))
    with pytest.raises(gate.GateFailure, match="sast_findings_require_triage"):
        gate.check_sarif(tmp_path)
    report["runs"][0] = {"results": [], "invocations": [{"executionSuccessful": False}]}
    path.write_text(json.dumps(report))
    with pytest.raises(gate.GateFailure, match="sast_execution_failed"):
        gate.check_sarif(tmp_path)


@pytest.mark.parametrize("exit_code", [0, 9])
def test_mend_retains_sanitized_findings_and_policy_outcome(gate, tmp_path, monkeypatch, exit_code):
    from subprocess import CompletedProcess

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEND_USER_KEY", "private-key")
    monkeypatch.setenv("MEND_SCOPE", "private-org//app//project")
    result = CompletedProcess(
        args=[],
        returncode=exit_code,
        stdout=json.dumps(
            {
                "scope": "private-org//app//project",
                "path": str(tmp_path / "dependencies"),
                "access_token": "another-private-token",
                "diagnostic": "Bearer private-key",
                "vulnerabilities": [{"id": "CVE-example", "severity": "HIGH", "package": "public"}],
            }
        ),
    )
    gate.retain_mend_evidence(result, version="1.2.3", digest="a" * 64)
    report = (tmp_path / "artifacts/mend-report.json").read_text()
    assert not any(
        value in report
        for value in ("private-key", "private-org", "another-private", str(tmp_path))
    )
    assert json.loads(report)["vulnerabilities"][0]["id"] == "CVE-example"
    identity = json.loads((tmp_path / "artifacts/mend-identity.json").read_text())
    assert identity == {"version": "1.2.3", "sha256": "a" * 64, "exit_code": exit_code}
