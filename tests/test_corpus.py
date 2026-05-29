from dwp.corpus import Document, search


def test_search_matches_keyword():
    results = search("OpenInference", k=3)
    assert results, "should return at least one document"
    assert results[0].id == "doc-openinference"


def test_search_fallback_no_match():
    # No overlap → should still return k docs (never empty)
    results = search("xyzzy nonsense gobbledygook", k=2)
    assert len(results) == 2


def test_search_k_limit():
    results = search("OpenInference tracing batch async", k=1)
    assert len(results) == 1
    assert isinstance(results[0], Document)


def test_search_returns_documents():
    results = search("Doubleword inference", k=2)
    for doc in results:
        assert isinstance(doc, Document)
        assert doc.id and doc.title and doc.text
