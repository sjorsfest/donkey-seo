"""Tests for scraper brand asset candidate extraction."""

from bs4 import BeautifulSoup

from app.integrations.scraper import WebsiteScraper


def test_extract_asset_candidates_detects_logo_meta_and_icon() -> None:
    scraper = WebsiteScraper()
    html = """
    <html>
      <head>
        <meta property="og:image" content="/og-image.png" />
        <meta name="twitter:image" content="https://cdn.example.com/twitter.png" />
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body>
        <img src="/assets/company-logo.svg" alt="Acme logo" />
      </body>
    </html>
    """
    soup = BeautifulSoup(html, "lxml")

    candidates = scraper._extract_asset_candidates(
        soup=soup,
        page_url="https://acme.test/about",
    )

    by_url = {item["url"]: item for item in candidates}

    assert "https://acme.test/og-image.png" in by_url
    assert by_url["https://acme.test/og-image.png"]["role"] == "hero"

    assert "https://cdn.example.com/twitter.png" in by_url
    assert by_url["https://cdn.example.com/twitter.png"]["origin"] == "meta_twitter_image"

    assert "https://acme.test/favicon.ico" in by_url
    assert by_url["https://acme.test/favicon.ico"]["role"] == "icon"

    assert "https://acme.test/assets/company-logo.svg" in by_url
    assert by_url["https://acme.test/assets/company-logo.svg"]["role"] == "logo"


def test_extract_asset_candidates_prefers_higher_confidence_per_url() -> None:
    scraper = WebsiteScraper()
    html = """
    <html>
      <head><meta property="og:image" content="/shared.png" /></head>
      <body><img src="/shared.png" class="hero-image" alt="hero" /></body>
    </html>
    """
    soup = BeautifulSoup(html, "lxml")

    candidates = scraper._extract_asset_candidates(
        soup=soup,
        page_url="https://acme.test/",
    )

    assert len(candidates) == 1
    assert candidates[0]["url"] == "https://acme.test/shared.png"
    assert candidates[0]["origin"] == "meta_og_image"
    assert candidates[0]["role_confidence"] == 0.9


def test_extract_visual_signals_captures_colors_typography_and_ctas() -> None:
    scraper = WebsiteScraper()
    html = """
    <html>
      <head>
        <style>
          :root { --brand-primary: #f4e08a; }
          .hero-title { font-family: "Baloo 2", sans-serif; color: #132646; }
          .cta { border-radius: 9999px; box-shadow: 0 10px 20px rgba(0,0,0,0.2); }
        </style>
      </head>
      <body>
        <h1>Support chat that lives in your Telegram Group</h1>
        <a class="cta rounded-pill" href="/signup">Start for free</a>
        <button class="border-2">See how it works</button>
        <img src="/images/donkey-mascot.png" alt="Donkey mascot illustration" />
      </body>
    </html>
    """
    soup = BeautifulSoup(html, "lxml")

    signals = scraper._extract_visual_signals(soup=soup)

    assert "#F4E08A" in signals["observed_hex_colors"]
    assert "Baloo 2" in signals["observed_font_families"]
    assert "Start for free" in signals["cta_labels"]
    assert "Support chat that lives in your Telegram Group" in signals["hero_headlines"]
    assert "Illustrative/mascot imagery" in signals["imagery_cues"]
