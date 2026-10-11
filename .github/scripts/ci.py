"""Local and CI release-evidence checks; no package publication."""

import argparse
import hashlib
import json
import os
import platform
import re
import sqlite3
import ssl
import subprocess
import sys
import urllib.request
from importlib.metadata import distributions
from pathlib import Path
from urllib.parse import urlsplit


class GateFailure(Exception):
    pass


def run(*arguments, **kwargs):
    subprocess.run(arguments, check=True, **kwargs)


def environment_python(directory):
    return str(directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))


def export_requirements(destination, *, development):
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(
        "uv",
        "export",
        "--frozen",
        "--no-emit-project",
        "--all-groups" if development else "--no-dev",
        "--format",
        "requirements-txt",
        "--output-file",
        str(destination),
        stdout=subprocess.DEVNULL,
    )


def inventory():
    directory = Path("artifacts")
    directory.mkdir(exist_ok=True)
    with sqlite3.connect(":memory:") as database:
        database.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
        database.execute("INSERT INTO probe VALUES('verified')")
        database.commit()
        image = database.serialize()
    with sqlite3.connect(":memory:") as database:
        database.deserialize(image)
        assert database.execute("SELECT text FROM probe").fetchone() == ("verified",)
    from loom.documents import embeddings

    evidence = {
        "python": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "sqlite": sqlite3.sqlite_version,
        "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "packages": sorted(
            ({"name": item.metadata["Name"], "version": item.version} for item in distributions()),
            key=lambda item: item["name"].lower(),
        ),
        "embedding": {
            "model": embeddings.MODEL,
            "repository": embeddings.REPOSITORY,
            "revision": embeddings.REVISION,
            "license": "MIT",
            "artifacts": {
                name: {"bytes": size, "sha256": digest}
                for name, (size, digest) in embeddings.ARTIFACTS.items()
            },
        },
    }
    (directory / "inventory.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")


def sbom(python, destination):
    from cyclonedx.schema import SchemaVersion
    from cyclonedx.validation.json import JsonStrictValidator

    run(
        "cyclonedx-py",
        "environment",
        python,
        "--spec-version",
        "1.6",
        "--output-format",
        "JSON",
        "--output-file",
        str(destination),
        "--validate",
    )
    value = json.loads(destination.read_text(encoding="utf-8"))

    def remove_local_references(node):
        if isinstance(node, dict):
            if "externalReferences" in node:
                node["externalReferences"] = [
                    reference
                    for reference in node["externalReferences"]
                    if urlsplit(reference["url"]).scheme != "file"
                ]
            for child in node.values():
                remove_local_references(child)
        elif isinstance(node, list):
            for child in node:
                remove_local_references(child)

    remove_local_references(value)
    encoded = json.dumps(value, indent=2)
    if str(Path.cwd()) in encoded or Path.cwd().as_posix() in encoded:
        raise GateFailure("private_path_in_sbom")
    if JsonStrictValidator(SchemaVersion.V1_6).validate_str(encoded) is not None:
        raise GateFailure("invalid_sanitized_sbom")
    destination.write_text(encoded, encoding="utf-8")


def package():
    inventory()
    run("uv", "build", "--no-build-isolation", "--out-dir", "dist")
    runtime = Path("dist/.runtime-requirements.txt")
    development = Path("dist/.development-requirements.txt")
    export_requirements(runtime, development=False)
    export_requirements(development, development=True)
    smoke = (
        "import importlib.metadata as m, pathlib, sqlite3, sys, loom;"
        "assert pathlib.Path(loom.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix));"
        "assert all(hasattr(loom, name) for name in loom.__all__);"
        "assert m.version('loom-agentic-framework') == loom.__version__;"
        "db=sqlite3.connect(':memory:');db.execute('CREATE VIRTUAL TABLE f USING fts5(text)');"
        "db.serialize();print('Installed distribution and SQLite features verified')"
    )
    for suffix, name, requirements in (
        ("*.whl", "wheel", runtime),
        ("*.tar.gz", "source", development),
    ):
        artifacts = list(Path("dist").glob(suffix))
        if len(artifacts) != 1:
            raise GateFailure("ambiguous_distribution")
        directory = Path(f"dist/.{name}-environment")
        run("uv", "venv", "--clear", "--python", sys.executable, str(directory))
        python = environment_python(directory)
        run("uv", "pip", "install", "--python", python, "--require-hashes", "-r", str(requirements))
        run(
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--no-build-isolation",
            "--no-deps",
            str(artifacts[0]),
        )
        run(python, "-I", "-c", smoke)
        run(python, "-I", "-m", "loom.cli", "--version")
        if name == "wheel":
            sbom(python, Path("artifacts/runtime.cdx.json"))
    sbom(sys.executable, Path("artifacts/development.cdx.json"))
    Path("artifacts/distributions.json").write_text(
        json.dumps(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in Path("dist").iterdir()
                if path.name.endswith((".whl", ".tar.gz"))
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def check_sarif(directory):
    paths = list(directory.glob("*.sarif"))
    if not paths:
        raise GateFailure("missing_sast_evidence")
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("version") != "2.1.0" or not report.get("runs"):
            raise GateFailure("invalid_sast_evidence")
        for result in report["runs"]:
            if any(
                not invocation.get("executionSuccessful", False)
                for invocation in result.get("invocations", [])
            ):
                raise GateFailure("sast_execution_failed")
            if result.get("results"):
                raise GateFailure("sast_findings_require_triage")


def retain_mend_evidence(result, *, version, digest):
    directory = Path("artifacts")
    directory.mkdir(exist_ok=True)
    (directory / "mend-identity.json").write_text(
        json.dumps({"version": version, "sha256": digest, "exit_code": result.returncode}),
        encoding="utf-8",
    )
    report = json.loads(result.stdout)
    if not isinstance(report, (dict, list)):
        raise GateFailure("invalid_mend_evidence")
    private = {
        os.environ.get(name, "")
        for name in ("MEND_URL", "MEND_EMAIL", "MEND_USER_KEY", "MEND_SCOPE")
    }
    private.update((str(Path.cwd()), Path.cwd().as_posix(), Path.cwd().as_uri()))

    def redact(value):
        if isinstance(value, str):
            for text in sorted(private - {""}, key=len, reverse=True):
                value = value.replace(text, "[REDACTED]")
            return value
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, dict):
            return {
                redact(key): "[REDACTED]"
                if re.search(
                    r"(?:^|[_-])(?:password|secret|token|api[_-]?key|user[_-]?key|authorization)$",
                    key,
                    flags=re.IGNORECASE,
                )
                else redact(item)
                for key, item in value.items()
            }
        return value

    (directory / "mend-report.json").write_text(
        json.dumps(redact(report), indent=2, allow_nan=False), encoding="utf-8"
    )


def mend():
    required = (
        "MEND_CLI_VERSION",
        "MEND_CLI_SHA256",
        "MEND_CLI_URL",
        "MEND_SCOPE",
        "MEND_URL",
        "MEND_EMAIL",
        "MEND_USER_KEY",
    )
    if any(not os.environ.get(name) for name in required):
        raise GateFailure("missing_approved_mend_inputs")
    version = os.environ["MEND_CLI_VERSION"]
    digest = os.environ["MEND_CLI_SHA256"]
    url = os.environ["MEND_CLI_URL"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise GateFailure("invalid_mend_identity")
    if urlsplit(url).scheme != "https" or urlsplit(url).username:
        raise GateFailure("invalid_mend_download_url")

    class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, message, headers, newurl):
            if urlsplit(newurl).scheme != "https" or urlsplit(newurl).username:
                raise GateFailure("insecure_mend_redirect")
            return super().redirect_request(request, fp, code, message, headers, newurl)

    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    opener = urllib.request.build_opener(
        HTTPSRedirect(), urllib.request.HTTPSHandler(context=context)
    )
    maximum = 256 * 1024**2
    with opener.open(url, timeout=120) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum or hashlib.sha256(data).hexdigest() != digest:
        raise GateFailure("mend_checksum_mismatch")
    directory = Path(".mend")
    directory.mkdir(mode=0o700, exist_ok=True)
    binary = directory / ("mend.exe" if os.name == "nt" else "mend")
    binary.write_bytes(data)
    binary.chmod(0o700)
    safe_environment = {
        key: value for key, value in os.environ.items() if not key.startswith("MEND_")
    }
    identity = subprocess.run(
        [str(binary.resolve()), "version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        env=safe_environment,
    )
    if not re.search(rf"(?<![\d.]){re.escape(version)}(?![\d.])", identity.stdout):
        raise GateFailure("mend_version_mismatch")
    stage = directory / "dependency-stage"
    stage.mkdir(exist_ok=True)
    export_requirements(stage / "requirements.txt", development=True)
    if {path.name for path in stage.iterdir()} != {"requirements.txt"}:
        raise GateFailure("non_dependency_scan_input")
    safe_environment.update(
        {name: os.environ[name] for name in ("MEND_URL", "MEND_EMAIL", "MEND_USER_KEY")}
    )
    result = subprocess.run(
        [
            str(binary.resolve()),
            "dep",
            "--dir",
            ".",
            "--dev",
            "--strict",
            "--fail-policy",
            "--format",
            "json",
            "--non-interactive",
            "--scope",
            os.environ["MEND_SCOPE"],
        ],
        cwd=stage,
        env=safe_environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode in {0, 9}:
        retain_mend_evidence(result, version=version, digest=digest)
    if result.returncode != 0:
        raise GateFailure("mend_policy_violation" if result.returncode == 9 else "mend_scan_failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("inventory", "package", "sarif", "mend"))
    args = parser.parse_args()
    try:
        if args.operation == "sarif":
            check_sarif(Path("artifacts/sarif"))
        else:
            {"inventory": inventory, "package": package, "mend": mend}[args.operation]()
    except (GateFailure, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(str(error) if isinstance(error, GateFailure) else "ci_gate_failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
