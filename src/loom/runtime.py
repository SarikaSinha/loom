from __future__ import annotations

import asyncio
import contextvars
import math
import os
import random
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Generic, TypeVar

from loom.contracts import CancellationToken, ToolFailure

T = TypeVar("T")


async def retry_pause(
    attempt: int,
    *,
    check: Callable[[], object],
    retry_after: object = None,
) -> None:
    delay = random.uniform(0, 0.5 * 2 ** (attempt - 1))
    if (
        isinstance(retry_after, (int, float))
        and not isinstance(retry_after, bool)
        and math.isfinite(retry_after)
        and retry_after >= 0
    ):
        delay = max(delay, retry_after)
    until = time.monotonic() + delay
    while True:
        check()
        remaining = until - time.monotonic()
        if remaining <= 0:
            return
        await asyncio.sleep(min(remaining, 0.05))


class Permit:
    def __init__(self, owner: WorkPool) -> None:
        self.owner = owner
        self._guard = threading.Lock()
        self._submitted = False
        self._released = False

    def take(self, owner: WorkPool) -> None:
        with self._guard:
            if owner is not self.owner or self._released or self._submitted:
                raise ValueError("invalid executor reservation")
            self._submitted = True

    def release(self) -> None:
        with self._guard:
            if not self._released:
                self._released = True
                self.owner._slots.release()


@dataclass(frozen=True)
class Work(Generic[T]):
    future: Future[T]
    abandoned: threading.Event


class WorkPool:
    def __init__(self, name: str, workers: int, queued: int) -> None:
        self.name = name
        self.capacity = workers + queued
        self._slots = threading.BoundedSemaphore(self.capacity)
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"loom-{name}")

    def reserve(self) -> Permit:
        if not self._slots.acquire(blocking=False):
            raise ToolFailure(
                "executor_busy",
                recoverable=True,
                details={"pool": self.name, "capacity": self.capacity},
            )
        return Permit(self)

    def submit(
        self,
        function: Callable[[], T],
        *,
        deadline: float,
        cancellation: CancellationToken | None = None,
        finished: Callable[[], None] | None = None,
        permit: Permit | None = None,
    ) -> Work[T]:
        permit = permit or self.reserve()
        permit.take(self)
        context = contextvars.copy_context()
        abandoned = threading.Event()

        def execute() -> T:
            try:
                if (
                    abandoned.is_set()
                    or time.monotonic() >= deadline
                    or (cancellation and cancellation.is_cancelled())
                ):
                    raise ToolFailure("operation_cancelled", recoverable=True)
                return context.run(function)
            finally:
                try:
                    if finished:
                        finished()
                finally:
                    permit.release()

        try:
            return Work(self._executor.submit(execute), abandoned)
        except RuntimeError:
            permit.release()
            raise ToolFailure("executor_unavailable") from None

    def close(self) -> None:
        self._executor.shutdown(wait=True)


class OwnershipLease:
    def __init__(self, manager: AbstractContextManager[object]) -> None:
        self.manager = manager
        self._guard = threading.Lock()
        self._references = 1
        self._entered = False

    def enter(self) -> None:
        self.manager.__enter__()
        with self._guard:
            self._entered = True
            abandoned = self._references == 0
        if abandoned:
            self.manager.__exit__(None, None, None)

    def retain(self) -> Callable[[], None]:
        with self._guard:
            if self._references == 0:
                raise RuntimeError("resource ownership has ended")
            self._references += 1
        released = False
        guard = threading.Lock()

        def release() -> None:
            nonlocal released
            with guard:
                if not released:
                    released = True
                    self.release()

        return release

    def release(self) -> None:
        with self._guard:
            self._references -= 1
            close = self._references == 0 and self._entered
            if close:
                self._entered = False
        if close:
            self.manager.__exit__(None, None, None)

    async def offload(
        self,
        name: str,
        function: Callable[[], T],
        *,
        seconds: float,
        cancellation: CancellationToken | None = None,
    ) -> T:
        executor = pool(name)
        permit = executor.reserve()
        release = self.retain()
        deadline = time.monotonic() + seconds
        try:
            work = executor.submit(
                function,
                permit=permit,
                deadline=deadline,
                cancellation=cancellation,
                finished=release,
            )
        except BaseException:
            permit.release()
            release()
            raise
        return await wait_work(work, deadline=deadline, cancellation=cancellation)


_guard = threading.Lock()
_pools: dict[str, WorkPool] = {}
_specs = {"user_code": (4, 0), "internal_io": (2, 8), "inference": (1, 2)}


def _after_fork() -> None:
    global _guard, _pools
    _guard, _pools = threading.Lock(), {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def pool(name: str) -> WorkPool:
    with _guard:
        if name not in _pools:
            workers, queued = _specs[name]
            _pools[name] = WorkPool(name, workers, queued)
        return _pools[name]


async def wait_work(
    work: Work[T],
    *,
    deadline: float,
    cancellation: CancellationToken | None = None,
) -> T:
    wrapped = asyncio.wrap_future(work.future)
    # Keep the concurrent future alive after its waiter is cancelled.
    protected = asyncio.shield(wrapped)
    try:
        while True:
            if cancellation and cancellation.is_cancelled():
                raise ToolFailure("operation_cancelled", recoverable=True)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            done, _ = await asyncio.wait({protected}, timeout=min(remaining, 0.05))
            if done:
                return protected.result()
    finally:
        if not work.future.done():
            work.abandoned.set()
        if not protected.done():
            protected.cancel()

        # Retrieve late exceptions without tying resource ownership to the event loop.
        def consume(done: asyncio.Future[T]) -> None:
            if not done.cancelled():
                done.exception()

        wrapped.add_done_callback(consume)


async def wait_async(
    operation: Awaitable[T],
    *,
    seconds: float,
    cancellation: CancellationToken | None = None,
    finished: Callable[[], None] | None = None,
) -> T:
    try:
        task = asyncio.ensure_future(operation)
    except BaseException:
        if finished:
            finished()
        raise

    def completed(future: asyncio.Future[T]) -> None:
        if finished:
            finished()
        if not future.cancelled():
            future.exception()

    task.add_done_callback(completed)
    deadline = time.monotonic() + seconds
    try:
        while True:
            if cancellation is not None and cancellation.is_cancelled():
                raise ToolFailure("operation_cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            ready, _ = await asyncio.wait({task}, timeout=min(remaining, 0.05))
            if ready:
                return task.result()
    finally:
        if not task.done():
            task.cancel()


async def offload(
    name: str,
    function: Callable[[], T],
    *,
    seconds: float,
    cancellation: CancellationToken | None = None,
    finished: Callable[[], None] | None = None,
) -> T:
    deadline = time.monotonic() + seconds
    future = pool(name).submit(
        function, deadline=deadline, cancellation=cancellation, finished=finished
    )
    return await wait_work(future, deadline=deadline, cancellation=cancellation)


def sync(factory: Callable[[], Coroutine[object, object, T]]) -> T:
    from loom.contracts import LoomConfigurationError

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    raise LoomConfigurationError("active_event_loop", "Use the asynchronous API in an event loop.")


@asynccontextmanager
async def ownership(manager: AbstractContextManager[object]) -> AsyncIterator[None]:
    async with leased_ownership(manager):
        yield


@asynccontextmanager
async def leased_ownership(
    manager: AbstractContextManager[object],
) -> AsyncIterator[OwnershipLease]:
    lease = OwnershipLease(manager)
    try:
        await offload("internal_io", lease.enter, seconds=30)
        yield lease
    finally:
        lease.release()
