"""Schema-level tests for content article contracts."""

import pytest

from app.schemas.content import ArticleBlock, ContentArticleResponse, RegenerateArticleRequest


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


def test_content_article_response_accepts_optional_author_id() -> None:
    article = ContentArticleResponse.model_validate(
        {
            "id": "article_1",
            "project_id": "project_1",
            "brief_id": "brief_1",
            "author_id": "author_1",
            "title": "Title",
            "slug": "title",
            "primary_keyword": "keyword",
            "status": "draft",
            "publish_status": None,
            "published_at": None,
            "published_url": None,
            "current_version": 1,
            "generation_model": None,
            "generated_at": "2026-02-25T10:00:00Z",
            "created_at": "2026-02-25T10:00:00Z",
            "updated_at": "2026-02-25T10:00:00Z",
        }
    )

    assert article.author_id == "author_1"
