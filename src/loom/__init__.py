"""Loom's embedded agent framework."""

__version__ = "0.1.0"

from loom.agent import Agent
from loom.config import Config
from loom.contracts import (
    AgentSpec,
    ApprovalDecision,
    ApprovalRequest,
    CallerIdentity,
    CancellationToken,
    ConversationState,
    Limits,
    LoomConfigurationError,
    LoomStorageError,
    RunResult,
    ToolContext,
    ToolFailure,
    ToolSpec,
    WorkflowLimits,
)
from loom.documents import DocumentStore
from loom.storage import JSONStorage, SQLiteStorage, Storage
from loom.tools.base import Tool
from loom.tools.http import HTTPTool
from loom.workflows import Branch, Case, Step, Workflow, WorkflowView

__all__ = [
    "Agent",
    "AgentSpec",
    "Tool",
    "HTTPTool",
    "DocumentStore",
    "Storage",
    "SQLiteStorage",
    "JSONStorage",
    "Config",
    "Limits",
    "WorkflowLimits",
    "Workflow",
    "WorkflowView",
    "Step",
    "Branch",
    "Case",
    "RunResult",
    "ConversationState",
    "CancellationToken",
    "ToolContext",
    "ToolFailure",
    "CallerIdentity",
    "ApprovalRequest",
    "ApprovalDecision",
    "ToolSpec",
    "LoomConfigurationError",
    "LoomStorageError",
]
