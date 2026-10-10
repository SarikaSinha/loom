import asyncio
import os

import numpy as np
import pytest

from loom import Config, ToolFailure
from loom.documents.embeddings import encoder

pytestmark = pytest.mark.integration


def prepared_config():
    path = os.environ.get("LOOM_EMBEDDING_CONFIG")
    assert path, "Set LOOM_EMBEDDING_CONFIG after explicitly running loom setup embeddings"
    return Config.load(path)


def test_real_pinned_embeddings_are_finite_normalized_and_reused():
    config = prepared_config()
    model = encoder(config)
    passages = model.embed(["API requests have a rate limit.", "Pagination returns more items."])
    query = model.embed(["What is the API rate limit?"], query=True)
    assert passages.shape == (2, 384) and query.shape == (1, 384)
    assert passages.dtype == query.dtype == np.float32
    assert np.isfinite(passages).all() and np.isfinite(query).all()
    np.testing.assert_allclose(np.linalg.norm(passages, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(query, axis=1), 1, atol=1e-6)

    async def load_again():
        return await asyncio.to_thread(encoder, config)

    assert asyncio.run(load_again()) is model
    assert asyncio.run(load_again()) is model


def test_real_tokenizer_rejects_oversized_queries():
    model = encoder(prepared_config())
    with pytest.raises(ToolFailure) as failure:
        model.embed(["overflow " * 600], query=True)
    assert failure.value.code == "query_too_long"
