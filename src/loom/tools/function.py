from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast, get_type_hints

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, create_model

from loom.contracts import (
    JSONValue,
    LoomConfigurationError,
    Schema,
    ToolContext,
    ToolFailure,
    ToolSpec,
    json_value,
    schema_for,
)
from loom.runtime import offload
from loom.tools.base import Tool

if TYPE_CHECKING:
    from typing_extensions import TypeForm


def annotation_schema(annotation: TypeForm[object]) -> Schema:
    try:
        schema = json_value(TypeAdapter(annotation).json_schema())
    except (TypeError, ValueError):
        raise LoomConfigurationError("unsupported_tool_annotation") from None
    if not isinstance(schema, dict):
        raise LoomConfigurationError("unsupported_tool_annotation")
    return schema


class FunctionTool(Tool):
    def __init__(
        self,
        fn: Callable[..., object],
        *,
        name: str | None,
        description: str | None,
        version: str,
        argument_schema: Schema | type[BaseModel] | None,
        output_schema: Schema | type[BaseModel] | None,
        sensitive: bool,
        retry_safe: bool,
        timeout_seconds: float | None,
        egress: Literal["none", "fixed_network", "unknown"],
        egress_origins: tuple[str, ...],
    ) -> None:
        signature = inspect.signature(fn)
        try:
            annotations: dict[str, TypeForm[object]] = get_type_hints(fn)
        except (NameError, TypeError):
            raise LoomConfigurationError("unsupported_tool_annotation") from None
        fields: dict[str, tuple[TypeForm[object], object]] = {}
        self._annotations: dict[str, TypeForm[object]] = {}
        self._takes_context = False
        for parameter in signature.parameters.values():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            ):
                raise LoomConfigurationError("unsupported_tool_parameter")
            annotation = annotations.get(parameter.name)
            if parameter.name == "context" and annotation is ToolContext:
                if parameter.kind != inspect.Parameter.KEYWORD_ONLY:
                    raise LoomConfigurationError("context_must_be_keyword_only")
                self._takes_context = True
                continue
            if annotation is None:
                raise LoomConfigurationError("missing_tool_annotation")
            self._annotations[parameter.name] = annotation
            fields[parameter.name] = (
                annotation,
                ... if parameter.default is inspect.Parameter.empty else parameter.default,
            )
        try:
            self.arguments_model = create_model(
                "ToolArguments",
                __config__=ConfigDict(strict=True, extra="forbid"),
                **cast(dict[str, Any], fields),
            )
        except (ValueError, TypeError):
            raise LoomConfigurationError("unsupported_tool_annotation") from None
        arguments: Schema | None = (
            schema_for(argument_schema)
            if argument_schema is not None
            else schema_for(self.arguments_model)
        )
        if output_schema is not None:
            output = schema_for(output_schema)
        elif "return" in annotations:
            output = annotation_schema(annotations["return"])
        else:
            raise LoomConfigurationError("missing_tool_return_annotation")
        if arguments is None or output is None:
            raise LoomConfigurationError("invalid_tool_schema")
        doc = description if description is not None else inspect.getdoc(fn)
        tool_name = name or getattr(fn, "__name__", "")
        if not doc or not tool_name:
            raise LoomConfigurationError("missing_tool_description")
        super().__init__(
            ToolSpec(
                name=tool_name,
                description=doc,
                version=version,
                argument_schema=arguments,
                output_schema=output,
                sensitive=sensitive,
                retry_safe=retry_safe,
                timeout_seconds=timeout_seconds,
                egress=egress,
                egress_origins=egress_origins,
            )
        )
        self.fn = fn
        self.is_async = inspect.iscoroutinefunction(fn)

    def invoke(self, arguments: Mapping[str, JSONValue], context: ToolContext) -> object:
        import json

        values: dict[str, object] = {}
        for key, value in arguments.items():
            annotation = self._annotations[key]
            values[key] = TypeAdapter(annotation).validate_json(
                json.dumps(value, allow_nan=False), strict=True
            )
        if self._takes_context:
            values["context"] = context
        return self.fn(**values)

    async def execute(
        self,
        arguments: Mapping[str, JSONValue],
        context: ToolContext,
    ) -> JSONValue | BaseModel:
        from datetime import UTC, datetime

        if self.is_async:
            value = self.invoke(arguments, context)
            if not inspect.isawaitable(value):
                raise LoomConfigurationError("invalid_async_tool")
            result = await value
        else:
            result = await offload(
                "user_code",
                lambda: self.invoke(arguments, context),
                seconds=max(0.001, (context.deadline - datetime.now(UTC)).total_seconds()),
                cancellation=context.cancellation,
            )
        try:
            return result if isinstance(result, BaseModel) else json_value(result)
        except ValidationError:
            raise ToolFailure("invalid_json_output") from None
