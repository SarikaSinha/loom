from collections.abc import Iterator, Sequence


def windows(text: str, offsets: Sequence[tuple[int, int]]) -> Iterator[tuple[int, int, str]]:
    for begin in range(0, len(offsets), 350):
        end = min(begin + 400, len(offsets))
        start_char, end_char = offsets[begin][0], offsets[end - 1][1]
        content = text[start_char:end_char]
        if not content.strip():
            continue
        yield start_char, end_char, content
        if end == len(offsets):
            break
