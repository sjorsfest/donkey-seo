"""Schema-level tests for content article contracts."""

import pytest

from app.schemas.content import ArticleBlock, RegenerateArticleRequest


def test_article_block_accepts_supported_type() -> None:
    block = ArticleBlock(
        block_type="faq",
        semantic_tag="section",
        heading="FAQ",
        faq_items=[{"question": "Q", "answer": "A"}],
    )

    assert block.block_type == "faq"
    assert block.faq_items[0].question == "Q"


def test_article_block_rejects_unknown_type() -> None:
    with pytest.raises(Exception):
        ArticleBlock(block_type="unknown", semantic_tag="section")


def test_regenerate_request_defaults() -> None:
    request = RegenerateArticleRequest.model_validate({})
    assert request.reason is None
