from uuid import uuid4

from app.rag.retrieval.citation_engine import (
    CitationEngine,
)
from app.rag.retrieval.context_builder import (
    BuiltContext,
    ContextSource,
)


def create_context_source(index: int):
    return ContextSource(
        context_index=index,
        chunk_id=f"chunk-{index}",
        document_id=uuid4(),
        rank=index,
        relevance_score=0.85,
        source_filename="engineering.pdf",
        page_number=index,
        section_title="Architecture",
        source_reference=f"Page {index}",
        included_characters=100,
        was_truncated=False,
    )


def create_built_context():
    sources = (
        create_context_source(1),
        create_context_source(2),
    )

    return BuiltContext(
        project_id=uuid4(),
        query="vehicle architecture",
        text="Retrieved engineering context",
        sources=sources,
        total_characters=1000,
        available_chunk_count=2,
        included_chunk_count=2,
        truncated_chunk_count=0,
        context_was_limited=False,
    )


def test_citation_engine_builds_citations():
    engine = CitationEngine()

    context = create_built_context()

    bundle = engine.build(context)

    assert bundle.count == 2
    assert bundle.citations[0].citation_id == "SOURCE 1"
    assert bundle.citations[1].citation_id == "SOURCE 2"


def test_citation_lookup_works():
    engine = CitationEngine()

    context = create_built_context()

    bundle = engine.build(context)

    citation = bundle.get("source 1")

    assert citation is not None
    assert citation.chunk_id == "chunk-1"


def test_citation_preserves_source_order():
    engine = CitationEngine()

    context = create_built_context()

    bundle = engine.build(context)

    assert (
        bundle.citations[0].source_index
        <
        bundle.citations[1].source_index
    )