from __future__ import annotations

import asyncio
import inspect
import os
import select
import sys
import threading
import time
from collections.abc import Awaitable, Callable

from loom.contracts import ApprovalDecision, ApprovalRequest, ToolContext
from loom.events import redact
from loom.runtime import OwnershipLease, offload, wait_async

ApprovalCallback = Callable[
    [ApprovalRequest, ToolContext], ApprovalDecision | Awaitable[ApprovalDecision]
]
_terminal_lock = threading.Lock()


async def approve(
    request: ApprovalRequest,
    context: ToolContext,
    callback: ApprovalCallback | None,
    seconds: float,
    *,
    lease: OwnershipLease | None = None,
) -> tuple[ApprovalDecision, str | None]:
    if callback is not None:
        result: ApprovalDecision | Awaitable[ApprovalDecision]
        try:
            deadline = time.monotonic() + seconds
            if inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(
                type(callback).__call__
            ):
                pending = callback(request, context)
                if not inspect.isawaitable(pending):
                    raise TypeError("approval callback must return an awaitable")
                result = await wait_async(
                    pending,
                    seconds=seconds,
                    cancellation=context.cancellation,
                    finished=lease.retain() if lease is not None else None,
                )
            else:
                result = await (lease.offload if lease is not None else offload)(
                    "user_code",
                    lambda: callback(request, context),
                    seconds=seconds,
                    cancellation=context.cancellation,
                )
                if inspect.isawaitable(result):
                    result = await wait_async(
                        result,
                        seconds=max(0, deadline - time.monotonic()),
                        cancellation=context.cancellation,
                        finished=lease.retain() if lease is not None else None,
                    )
            if not isinstance(result, ApprovalDecision):
                return ApprovalDecision(approved=False, reason="invalid callback result"), (
                    "approval_invalid_result"
                )
            return result, None
        except Exception as error:
            from loom.contracts import ToolFailure

            code = (
                "approval_executor_busy"
                if isinstance(error, ToolFailure) and error.code == "executor_busy"
                else "approval_callback_failed"
            )
            return ApprovalDecision(approved=False, reason=code), code
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        return ApprovalDecision(approved=False, reason="headless"), None
    deadline = time.monotonic() + seconds
    while not _terminal_lock.acquire(blocking=False):
        if context.cancellation.is_cancelled() or time.monotonic() >= deadline:
            return ApprovalDecision(approved=False, reason="approval_timeout"), None
        await asyncio.sleep(0.02)
    try:
        print(
            f"loom: approve {request.tool_name} for {context.caller.subject}; "
            f"contains document content={request.contains_document_content}; "
            f"arguments={redact(request.arguments)} [yes/no]",
            file=sys.stderr,
            flush=True,
        )
        answer = ""
        while time.monotonic() < deadline and not context.cancellation.is_cancelled():
            char: str | None = None
            if sys.platform == "win32":
                import msvcrt

                if msvcrt.kbhit():
                    char = msvcrt.getwch()
            elif select.select([sys.stdin], [], [], 0)[0]:
                char = os.read(sys.stdin.fileno(), 1).decode("utf-8", errors="replace")
            if char in {"", "\r", "\n"}:
                approved = answer.lower() == "yes"
                return ApprovalDecision(
                    approved=approved, reason="" if approved else "terminal_denied"
                ), None
            if char is not None:
                answer += char
                if len(answer) > 16:
                    return ApprovalDecision(approved=False, reason="terminal_denied"), None
            await asyncio.sleep(0.02)
        return ApprovalDecision(approved=False, reason="approval_timeout"), None
    finally:
        _terminal_lock.release()
