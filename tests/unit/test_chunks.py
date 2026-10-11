import pytest

from loom.documents.chunks import windows


@pytest.mark.parametrize(
    "tokens,bounds",
    [
        (0, []),
        (399, [(0, 399)]),
        (400, [(0, 400)]),
        (401, [(0, 400), (350, 401)]),
        (750, [(0, 400), (350, 750)]),
        (751, [(0, 400), (350, 750), (700, 751)]),
    ],
)
def test_windows_preserve_exact_token_bounds_and_overlap(tokens, bounds):
    text = "w " * tokens
    offsets = [(index * 2, index * 2 + 1) for index in range(tokens)]
    chunks = list(windows(text, offsets))
    assert [(start // 2, (end + 1) // 2) for start, end, _ in chunks] == bounds
    assert all(content == text[start:end] for start, end, content in chunks)
