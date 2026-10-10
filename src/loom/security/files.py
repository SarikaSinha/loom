from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import portalocker

from loom.contracts import LoomStorageError

_locks_guard = threading.Lock()
_locks: dict[str, _PathLock] = {}


def _after_fork() -> None:
    global _locks_guard, _locks
    _locks_guard, _locks = threading.Lock(), {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


class _PathLock:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.readers = 0
        self.writer = False

    def acquire(self, shared: bool, blocking: bool, deadline: float) -> bool:
        with self.condition:
            while self.writer or (not shared and self.readers):
                remaining = deadline - time.monotonic()
                if not blocking or remaining <= 0:
                    return False
                self.condition.wait(remaining)
            if shared:
                self.readers += 1
            else:
                self.writer = True
            return True

    def release(self, shared: bool) -> None:
        with self.condition:
            if shared:
                self.readers -= 1
            else:
                self.writer = False
            self.condition.notify_all()


def canonical_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path)).replace("\\", "/")


def reject_links(path: Path) -> None:
    for item in (path, *path.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise LoomStorageError("unsafe_artifact_path")


def check_permissions(path: Path, *, directory: bool = False) -> None:
    reject_links(path)
    if sys.platform == "win32":
        from loom.security.windows import check

        check(path)
        return
    info = path.stat()
    expected = 0o700 if directory else 0o600
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != expected:
        raise LoomStorageError("insecure_permissions")
    if sys.platform == "darwin":
        result = subprocess.run(
            ["/bin/ls", "-lde", str(path)], capture_output=True, text=True, check=False
        )
        if result.returncode or len(result.stdout.splitlines()) != 1:
            raise LoomStorageError("insecure_permissions")


def secure_directory(path: Path) -> None:
    reject_links(path)
    missing: list[Path] = []
    parent = path
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for item in reversed(missing):
        try:
            if sys.platform == "win32":
                from loom.security.windows import create

                create(item, directory=True)
            else:
                item.mkdir(mode=0o700)
        except FileExistsError:
            check_permissions(item, directory=True)
    check_permissions(path, directory=True)


def private_open(path: Path, *, exclusive: bool = False) -> int:
    secure_directory(path.parent)
    reject_links(path)
    if path.exists():
        check_permissions(path)
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    flags |= os.O_EXCL if exclusive else os.O_APPEND
    if sys.platform == "win32":
        from loom.security.windows import create

        fd = create(path, exclusive=exclusive)
        assert fd is not None
    else:
        fd = os.open(path, flags, 0o600)
    try:
        check_permissions(path)
    except BaseException:
        os.close(fd)
        raise
    return fd


@contextmanager
def temporary_directory(parent: Path, *, prefix: str) -> Iterator[Path]:
    path = parent / f"{prefix}{uuid.uuid4().hex}"
    secure_directory(path)
    try:
        yield path
    finally:
        if path.exists():
            reject_links(path)
            shutil.rmtree(path)


def sync_directory(path: Path) -> None:
    if sys.platform != "win32":
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write(path: Path, data: bytes) -> None:
    secure_directory(path.parent)
    if path.exists():
        check_permissions(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = private_open(temporary, exclusive=True)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        import time

        for attempt in range(3):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if sys.platform != "win32" or attempt == 2:
                    raise
                time.sleep(0.05 * (attempt + 1))
        sync_directory(path.parent)
    except OSError:
        raise LoomStorageError("storage_write_failed") from None
    finally:
        temporary.unlink(missing_ok=True)


def read_private(path: Path, maximum: int) -> bytes:
    check_permissions(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        if os.fstat(stream.fileno()).st_size > maximum:
            raise LoomStorageError("artifact_too_large")
        result = stream.read(maximum + 1)
    if len(result) > maximum:
        raise LoomStorageError("artifact_too_large")
    return result


@contextmanager
def file_lock(
    path: Path,
    *,
    shared: bool = False,
    blocking: bool = True,
    deadline: float | None = None,
) -> Iterator[None]:
    deadline = deadline if deadline is not None else time.monotonic() + 30
    key = canonical_path(path)
    with _locks_guard:
        local = _locks.setdefault(key, _PathLock())
    if not local.acquire(shared, blocking, deadline):
        raise LoomStorageError("storage_lock_timeout" if blocking else "resource_in_use")
    fd: int | None = None
    try:
        fd = private_open(path)
        stream = os.fdopen(fd, "ab")
        fd = None
        with stream:
            flags = portalocker.LOCK_SH if shared else portalocker.LOCK_EX
            flags |= portalocker.LOCK_NB
            while True:
                try:
                    portalocker.lock(stream, flags)
                    break
                except portalocker.LockException:
                    remaining = deadline - time.monotonic()
                    if not blocking or remaining <= 0:
                        raise LoomStorageError(
                            "storage_lock_timeout" if blocking else "resource_in_use"
                        ) from None
                    time.sleep(min(0.01, remaining))
            try:
                yield
            finally:
                portalocker.unlock(stream)
    finally:
        if fd is not None:
            os.close(fd)
        local.release(shared)
