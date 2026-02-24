"""Tests for deferred internal link resolution on publication updates."""

from app.services.internal_link_resolver import replace_target_brief_links_in_document


def test_replace_target_brief_links_in_document_updates_matching_links() -> None:
    document = {
        "schema_version": "1.0",
        "blocks": [
            {
                "block_type": "section",
                "links": [
                    {"anchor": "A", "href": "/old-a", "target_brief_id": "brief-a"},
                    {"anchor": "B", "href": "/old-b", "target_brief_id": "brief-b"},
                ],
            }
        ],
    }

    updated, replacements = replace_target_brief_links_in_document(
        document=document,
        target_brief_id="brief-a",
        published_url="https://example.com/a",
    )

    assert replacements == 1
    links = updated["blocks"][0]["links"]
    assert links[0]["href"] == "https://example.com/a"
    assert links[1]["href"] == "/old-b"


def test_replace_target_brief_links_in_document_keeps_non_matching_links() -> None:
    document = {
        "schema_version": "1.0",
        "blocks": [
            {
                "block_type": "section",
                "links": [
                    {"anchor": "A", "href": "/old-a", "target_brief_id": "brief-a"},
                ],
                "cta": {"label": "Go", "href": "/old-cta", "target_brief_id": "brief-z"},
            }
        ],
    }

    updated, replacements = replace_target_brief_links_in_document(
        document=document,
        target_brief_id="brief-b",
        published_url="https://example.com/b",
    )

    assert replacements == 0
    assert updated["blocks"][0]["links"][0]["href"] == "/old-a"
    assert updated["blocks"][0]["cta"]["href"] == "/old-cta"
