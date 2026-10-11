from __future__ import annotations

from collections.abc import Mapping, Sequence

from loom.contracts import Provenance, ToolFailure, ToolSpec


def check_tool_egress(
    spec: ToolSpec,
    provenance: Sequence[Provenance],
    policies: Mapping[str, str],
    grants: Mapping[str, Sequence[str]],
) -> None:
    if not provenance or spec.egress == "none":
        return
    stores = {source.store for source in provenance}
    if spec.egress == "unknown" or any(
        policies.get(store) != "allow_hosted" or store not in grants.get(spec.name, ())
        for store in stores
    ):
        raise ToolFailure("document_egress_denied", recoverable=True)


def merge_provenance(*groups: Sequence[Provenance]) -> tuple[Provenance, ...]:
    merged: dict[tuple[str, str], set[str]] = {}
    for group in groups:
        for source in group:
            merged.setdefault((source.store, source.snapshot_id), set()).update(source.document_ids)
    return tuple(
        Provenance(store=store, snapshot_id=snapshot, document_ids=tuple(sorted(documents)))
        for (store, snapshot), documents in sorted(merged.items())
    )
