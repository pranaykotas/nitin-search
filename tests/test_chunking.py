from indexing.chunking import sub_chunk


def test_short_text_returned_as_single_chunk():
    text = "Just a short paragraph."
    assert sub_chunk(text) == [text]


def test_long_text_is_split_at_paragraph_boundaries():
    para = " ".join(["word"] * 300)
    text = f"{para}\n\n{para}\n\n{para}"

    chunks = sub_chunk(text, max_words=400, overlap_words=50)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.split()) <= 400 + 50


def test_consecutive_chunks_share_overlap_words():
    para = " ".join([f"word{i}" for i in range(300)])
    text = f"{para}\n\n{para}"

    chunks = sub_chunk(text, max_words=300, overlap_words=50)

    assert len(chunks) == 2
    first_tail = chunks[0].split()[-50:]
    second_head = chunks[1].split()[:50]
    assert first_tail == second_head


def test_single_giant_paragraph_is_split_with_word_level_chunks():
    """Test that a single oversized paragraph (no paragraph breaks) is split correctly.

    This tests the word-level fallback: when a single paragraph exceeds max_words,
    it must be split into fixed-size chunks with overlap, not buffered entirely.
    """
    para = " ".join([f"word{i}" for i in range(1000)])
    text = para  # No \n\n breaks — single giant paragraph

    chunks = sub_chunk(text, max_words=400, overlap_words=50)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.split()) <= 400 + 50
