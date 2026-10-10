from collections.abc import Sequence

from loom.contracts import (
    DocumentAnswer,
    JSONValue,
    SearchResult,
    ToolFailure,
    canonical,
    json_value,
)
from loom.documents.loaders import normalize


def validate_citations(value: JSONValue, searches: Sequence[SearchResult]) -> JSONValue:
    if not searches:
        raise ToolFailure("search_required")
    answer = DocumentAnswer.model_validate_json(canonical(value))
    seen: set[bytes] = set()
    citations = []
    for citation in answer.sources:
        excerpt = normalize(citation.excerpt)
        if not excerpt:
            raise ToolFailure("invalid_citation")
        citation = citation.model_copy(update={"excerpt": excerpt})
        found = any(
            source.store == citation.store
            and source.snapshot_id == citation.snapshot_id
            and chunk.chunk_id == citation.chunk_id
            and chunk.path == citation.path
            and chunk.location == citation.location
            and citation.excerpt in chunk.text
            for source in searches
            for chunk in source.chunks
        )
        if not found:
            raise ToolFailure("invalid_citation")
        key = canonical(citation)
        if key not in seen:
            seen.add(key)
            citations.append(citation)
    return json_value(answer.model_copy(update={"sources": citations}))
