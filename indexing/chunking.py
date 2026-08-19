from __future__ import annotations

MAX_CHUNK_WORDS = 400
OVERLAP_WORDS = 50


def sub_chunk(text: str, max_words: int = MAX_CHUNK_WORDS, overlap_words: int = OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping sub-chunks by word count.

    Short texts are returned as-is. Longer texts are split at paragraph
    boundaries where possible. Any paragraph exceeding max_words is itself
    split into fixed-size word-count chunks with word-level overlap, then
    treated as a complete unit in the output. This ensures every chunk is
    bounded by max_words (with overlap extending to max_words + overlap_words).
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    paragraphs = text.split("\n\n")
    sub_chunks: list[str] = []
    current_words: list[str] = []

    for para in paragraphs:
        para_words = para.split()

        # If this paragraph itself exceeds max_words, split it with word-level chunks
        if len(para_words) > max_words:
            # First, flush any accumulated words from previous paragraphs
            if current_words:
                sub_chunks.append(" ".join(current_words))
                current_words = []

            # Split the oversized paragraph into word-bounded chunks with overlap
            i = 0
            while i < len(para_words):
                chunk_words = para_words[i:i + max_words]
                sub_chunks.append(" ".join(chunk_words))
                i += max_words - overlap_words
        else:
            # Regular paragraph accumulation logic
            if current_words and len(current_words) + len(para_words) > max_words:
                sub_chunks.append(" ".join(current_words))
                overlap = current_words[-overlap_words:] if len(current_words) > overlap_words else []
                current_words = overlap + para_words
            else:
                current_words.extend(para_words)

    if current_words:
        sub_chunks.append(" ".join(current_words))

    return sub_chunks if sub_chunks else [text]
