"""Tests for the Retriever class.

These tests verify that the Retriever can be instantiated with a list of
documents, that it returns the most relevant documents for a query using
BM25 ranking, and that it handles edge‑cases such as empty document lists or
queries that contain no matching terms.
"""

import pytest

# The import path may vary depending on the project layout.
# We try the most common locations.
try:
    from retriever import Retriever  # type: ignore
except ImportError:  # pragma: no cover
    from src.retriever import Retriever  # type: ignore


@pytest.fixture
def sample_docs():
    """A small deterministic corpus used by the tests."""
    return [
        "apple banana cherry",
        "banana orange mango",
        "apple orange grape",
        "strawberry blueberry raspberry",
        "kiwi banana apple",
    ]


def test_retriever_initialization(sample_docs):
    """Retriever should accept a non‑empty list of documents."""
    retriever = Retriever(sample_docs)
    # The internal index should have the same number of documents.
    assert hasattr(retriever, "documents")
    assert len(retriever.documents) == len(sample_docs)


def test_retriever_empty_corpus_raises():
    """Creating a Retriever with no documents should raise a ValueError."""
    with pytest.raises(ValueError):
        Retriever([])


def test_retrieval_returns_correct_number_of_results(sample_docs):
    """The retrieve method should return at most `k` results."""
    retriever = Retriever(sample_docs)
    results = retriever.retrieve("apple", k=3)
    # The concrete return type may be a list of strings or a list of tuples.
    assert isinstance(results, list)
    assert len(results) <= 3
    # Ensure each element is either a string or a tuple/list with a string as first item.
    for item in results:
        if isinstance(item, (list, tuple)):
            assert isinstance(item[0], str)
        else:
            assert isinstance(item, str)


def test_bm25_ranking_prioritises_relevant_documents(sample_docs):
    """Documents containing the query term should be ranked higher."""
    retriever = Retriever(sample_docs)
    results = retriever.retrieve("apple", k=5)

    # Extract the document texts from the result regardless of the format.
    if results and isinstance(results[0], (list, tuple)):
        docs = [r[0] for r in results]
    else:
        docs = results

    # All documents that contain the word "apple" should appear before any that do not.
    apple_docs = [doc for doc in docs if "apple" in doc.lower()]
    non_apple_docs = [doc for doc in docs if "apple" not in doc.lower()]

    # There must be at least one apple document in the corpus.
    assert any("apple" in d.lower() for d in sample_docs)

    # The ordering constraint: every apple doc should precede any non‑apple doc.
    if non_apple_docs:
        first_non_apple_index = docs.index(non_apple_docs[0])
        last_apple_index = max(docs.index(d) for d in apple_docs) if apple_docs else -1
        assert last_apple_index < first_non_apple_index


def test_query_with_no_matching_terms_returns_empty_or_low_score(sample_docs):
    """A query that matches none of the documents should return an empty list
    or a list of documents with very low relevance scores."""
    retriever = Retriever(sample_docs)
    results = retriever.retrieve("pineapple", k=3)

    # If the implementation returns documents regardless of relevance, they should have
    # a score attribute that is close to zero. We handle both possibilities.
    if not results:
        assert results == []
    else:
        # Extract scores if present.
        if isinstance(results[0], (list, tuple)):
            scores = [r[1] for r in results if len(r) > 1]
            # Scores from BM25 are non‑negative; a very low score indicates no match.
            assert all(s < 0.1 for s in scores)
        else:
            # When only documents are returned, we cannot assert scores;
            # ensure that the returned docs do not contain the query term.
            assert all("pineapple" not in doc.lower() for doc in results)
