import asyncio
import os
import ssl
import threading
import time

import pytest
from pydantic import ValidationError

from loom import Config, Limits, LoomConfigurationError, LoomStorageError, ToolFailure
from loom.contracts import canonical
from loom.events import redact
from loom.runtime import WorkPool, wait_work
from loom.security.crypto import Crypto, resource_id
from loom.security.files import atomic_write, file_lock, read_private
from loom.security.transport import ssl_context, validate_url


@pytest.mark.parametrize(
    "values",
    [
        {"max_model_turns": True},
        {"max_tool_attempts": 0},
        {"deadline_seconds": float("inf")},
        {"max_retries": 3},
        {"max_corrections": 2},
        {"max_observation_bytes": 100},
        {"model_timeout_seconds": None},
        {"unknown": 1},
    ],
)
def test_strict_limits(values):
    with pytest.raises(ValidationError):
        Limits(**values)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "http://localhost:11434",
        "https://user:pass@example.com",
        "https://example.com/#fragment",
        "http://127.0.0.1.example.com",
    ],
)
def test_endpoint_rejections(url):
    with pytest.raises(LoomConfigurationError):
        validate_url(url, ollama=True)


def test_verified_tls_and_loopback():
    context = ssl_context(Config())
    assert context.minimum_version >= ssl.TLSVersion.TLSv1_2
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    assert validate_url("http://127.0.0.1:11434", ollama=True)
    assert validate_url("http://[::1]:11434", ollama=True)
    with pytest.raises(LoomConfigurationError):
        validate_url("http://127.0.0.1")


def test_canonical_bounds():
    assert canonical({"z": "é", "a": 1}) == b'{"a":1,"z":"\xc3\xa9"}'
    with pytest.raises(ToolFailure):
        canonical("x" * 5000, max_bytes=1024)
    with pytest.raises((ValueError, ValidationError)):
        canonical(float("nan"))


def test_redaction_is_exact():
    result = redact(
        {
            "access_token": "secret",
            "refreshToken": "secret",
            "X-API-Key": "secret",
            "max_output_tokens": 2048,
            "input_tokens": 12,
            "source_path": "private/file",
            "nested": {"password": "secret"},
        }
    )
    assert result == {
        "access_token": "[REDACTED]",
        "refreshToken": "[REDACTED]",
        "X-API-Key": "[REDACTED]",
        "max_output_tokens": 2048,
        "input_tokens": 12,
        "nested": {"password": "[REDACTED]"},
    }


def test_encryption_binding_and_tampering(tmp_path, encrypted_config):
    path = tmp_path / "data" / "state.enc"
    crypto = Crypto(encrypted_config)
    crypto.write(path, "storage_json", b'{"secret":"local document"}', 1)
    envelope = path.read_bytes()
    assert b"local document" not in envelope
    assert crypto.read(path, "storage_json", 1024)[1] == b'{"secret":"local document"}'
    altered = envelope[:-1] + bytes([envelope[-1] ^ 1])
    for data, identity in ((altered, resource_id(path, "storage_json")), (envelope, "0" * 64)):
        with pytest.raises(LoomStorageError):
            crypto.open(data, kind="storage_json", identity=identity, maximum=1024)
    with pytest.raises(LoomStorageError):
        crypto.seal(b"{}", kind="storage_json", identity="0" * 64, revision=1, run_id="stray")
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700


def test_atomic_failure_preserves_previous(tmp_path, monkeypatch):
    path = tmp_path / "private" / "state.enc"
    atomic_write(path, b"ciphertext")

    def fail(*args):
        raise OSError("simulated failure")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(LoomStorageError):
        atomic_write(path, b"new ciphertext")
    assert read_private(path, 1024) == b"ciphertext"
    assert list(path.parent.iterdir()) == [path]


def test_path_lock_shared_and_exclusive(tmp_path):
    path = tmp_path / "locks" / "read.lock"
    with file_lock(path, shared=True, blocking=False):
        with file_lock(path, shared=True, blocking=False):
            with pytest.raises(LoomStorageError):
                with file_lock(path, blocking=False):
                    pytest.fail("exclusive lock admitted with readers")


async def test_worker_timeout_keeps_capacity_and_cancelled_queue_never_runs():
    pool = WorkPool("test", 1, 1)
    release = threading.Event()
    entered = threading.Event()
    ran_queued = threading.Event()

    def blocked():
        entered.set()
        release.wait(5)

    try:
        first = pool.submit(blocked, deadline=time.monotonic() + 10)
        for _ in range(1000):
            if entered.is_set():
                break
            await asyncio.sleep(0.001)
        assert entered.is_set()
        second = pool.submit(ran_queued.set, deadline=time.monotonic() + 10)
        waiter = asyncio.create_task(wait_work(second, deadline=time.monotonic() + 10))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        with pytest.raises(ToolFailure, match="executor busy"):
            pool.submit(lambda: None, deadline=time.monotonic() + 10)
        with pytest.raises(TimeoutError):
            await wait_work(first, deadline=time.monotonic() + 0.01)
        release.set()
    finally:
        release.set()
        pool.close()
    assert not ran_queued.is_set()


@pytest.mark.skipif(os.name != "nt", reason="Windows security descriptor contract")
def test_windows_artifacts_are_private_at_creation(tmp_path):
    import subprocess

    from loom.security.files import secure_directory
    from loom.security.windows import check, create

    parent = tmp_path / "private"
    secure_directory(parent)
    directory = parent / "created"
    create(directory, directory=True)
    check(directory)
    path = directory / "state.enc"
    fd = create(path, exclusive=True)
    try:
        check(path)
        os.write(fd, b"ciphertext")
    finally:
        os.close(fd)
    subprocess.run(
        ["icacls", str(path), "/grant", "*S-1-1-0:(R)"],
        check=True,
        capture_output=True,
    )
    with pytest.raises(LoomStorageError, match="insecure permissions"):
        check(path)


@pytest.mark.skipif(os.name != "nt", reason="Windows replacement sharing contract")
def test_windows_open_reader_preserves_previous_file_on_replace_failure(tmp_path):
    import ctypes
    from ctypes import wintypes

    path = tmp_path / "private" / "state.enc"
    atomic_write(path, b"old ciphertext")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value
    try:
        with pytest.raises(LoomStorageError, match="storage write failed"):
            atomic_write(path, b"new ciphertext")
        assert path.read_bytes() == b"old ciphertext"
    finally:
        kernel.CloseHandle(handle)
    atomic_write(path, b"new ciphertext")
    assert read_private(path, 1024) == b"new ciphertext"
