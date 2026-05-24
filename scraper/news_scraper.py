"""
scraper/news_scraper.py
=======================
Web scraper untuk mengambil isi artikel berita dari berbagai
portal berita Indonesia (detik.com, kompas.com, tribunnews.com, dll.)
menggunakan requests + BeautifulSoup.
"""

import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Optional
import re


# ── HTTP Headers ──────────────────────────────────────────────────────────────
# Diperlukan agar server tidak menolak request kita sebagai bot
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

TIMEOUT = 10  # detik


# ── Data class hasil scraping ──────────────────────────────────────────────────
@dataclass
class ArticleResult:
    """Menyimpan hasil scraping satu artikel berita."""
    url: str
    title: str
    content: str
    source: str          # nama portal, misal "detik.com"
    success: bool
    error_message: Optional[str] = None


# ── Fungsi utama ───────────────────────────────────────────────────────────────
def scrape_article(url: str) -> ArticleResult:
    """
    Fungsi utama: ambil artikel dari URL apapun.
    Secara otomatis mendeteksi portal dan memilih parser yang tepat.

    Args:
        url: URL artikel berita (detik.com, kompas.com, dll.)

    Returns:
        ArticleResult berisi judul, konten, dan status.
    """
    source = _detect_source(url)

    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        # Pilih parser sesuai portal
        if "detik.com" in source:
            return _parse_detik(url, soup, source)
        elif "kompas.com" in source:
            return _parse_kompas(url, soup, source)
        elif "tribunnews.com" in source:
            return _parse_tribun(url, soup, source)
        elif "tempo.co" in source:
            return _parse_tempo(url, soup, source)
        else:
            # Generic parser untuk portal lain
            return _parse_generic(url, soup, source)

    except requests.exceptions.Timeout:
        return ArticleResult(
            url=url, title="", content="", source=source,
            success=False, error_message="Koneksi timeout. Coba lagi."
        )
    except requests.exceptions.ConnectionError:
        return ArticleResult(
            url=url, title="", content="", source=source,
            success=False, error_message="Gagal terhubung ke server."
        )
    except Exception as e:
        return ArticleResult(
            url=url, title="", content="", source=source,
            success=False, error_message=f"Error tidak terduga: {str(e)}"
        )


# ── Parser per portal ──────────────────────────────────────────────────────────

def _parse_detik(url: str, soup: BeautifulSoup, source: str) -> ArticleResult:
    """Parser khusus detik.com."""
    title = _get_text(soup, ["h1.detail__title", "h1.title", "h1"])
    content_div = soup.find("div", class_="detail__body-text") or \
                  soup.find("div", class_="itp_bodycontent")
    content = _extract_paragraphs(content_div)
    return _build_result(url, title, content, source)


def _parse_kompas(url: str, soup: BeautifulSoup, source: str) -> ArticleResult:
    """Parser khusus kompas.com."""
    title = _get_text(soup, ["h1.read__title", "h1"])
    content_div = soup.find("div", class_="read__content") or \
                  soup.find("div", attrs={"itemprop": "articleBody"})
    content = _extract_paragraphs(content_div)
    return _build_result(url, title, content, source)


def _parse_tribun(url: str, soup: BeautifulSoup, source: str) -> ArticleResult:
    """Parser khusus tribunnews.com."""
    title = _get_text(soup, ["h1.f-none", "h1"])
    content_div = soup.find("div", class_="side-article txt-article")
    content = _extract_paragraphs(content_div)
    return _build_result(url, title, content, source)


def _parse_tempo(url: str, soup: BeautifulSoup, source: str) -> ArticleResult:
    """Parser khusus tempo.co."""
    title = _get_text(soup, ["h1.title", "h1.text-32", "h1"])
    
    # Coba berbagai selector untuk konten (Tempo sering update layout)
    content_div = (
        soup.find("div", class_="detail-konten") or
        soup.find("section", class_="detail-konten") or
        soup.find("div", class_="detail-in") or
        soup.find("div", class_="article-content") or
        soup.find("article") or
        soup.find("div", attrs={"itemprop": "articleBody"})
    )
    content = _extract_paragraphs(content_div)
    
    # Kalau masih kosong, fallback ke generic parser
    if not content or len(content) < 100:
        return _parse_generic(url, soup, source)
    
    return _build_result(url, title, content, source)


def _parse_generic(url: str, soup: BeautifulSoup, source: str) -> ArticleResult:
    """
    Generic parser untuk portal yang belum punya parser khusus.
    Mengambil semua tag <p> di halaman dan memfilter yang cukup panjang.
    """
    title = _get_text(soup, ["h1"])

    # Ambil semua paragraf, filter yang isinya panjang (bukan menu/nav)
    paragraphs = soup.find_all("p")
    content_parts = [
        p.get_text(strip=True)
        for p in paragraphs
        if len(p.get_text(strip=True)) > 60
    ]
    content = "\n\n".join(content_parts)
    return _build_result(url, title, content, source)


# ── Helper functions ───────────────────────────────────────────────────────────

def _detect_source(url: str) -> str:
    """Deteksi nama portal dari URL."""
    url_lower = url.lower()
    portals = [
        "detik.com", "kompas.com", "tribunnews.com",
        "tempo.co", "cnnindonesia.com", "liputan6.com",
        "antaranews.com", "sindonews.com"
    ]
    for portal in portals:
        if portal in url_lower:
            return portal
    # Fallback: ambil domain dari URL
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1) if match else "unknown"


def _get_text(soup: BeautifulSoup, selectors: list) -> str:
    """Coba beberapa CSS selector, return teks pertama yang ditemukan."""
    for selector in selectors:
        el = soup.select_one(selector)
        if el:
            return el.get_text(strip=True)
    return "Judul tidak ditemukan"


def _extract_paragraphs(container) -> str:
    """Ambil semua teks <p> dari container HTML."""
    if not container:
        return ""
    paragraphs = container.find_all("p")
    texts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
    return "\n\n".join(texts)


def _build_result(url: str, title: str, content: str, source: str) -> ArticleResult:
    """Validasi hasil dan buat ArticleResult."""
    if not content or len(content) < 100:
        return ArticleResult(
            url=url, title=title, content=content, source=source,
            success=False,
            error_message="Konten artikel terlalu pendek atau tidak ditemukan. "
                          "Portal ini mungkin belum didukung sepenuhnya."
        )
    return ArticleResult(
        url=url, title=title, content=content,
        source=source, success=True
    )


# ── Quick test (jalankan file ini langsung) ────────────────────────────────────
if __name__ == "__main__":
    test_url = "https://www.detik.com/edu/edutainment/d-7296377/apa-itu-nlp"
    print(f"Testing scraper dengan URL: {test_url}\n")

    result = scrape_article(test_url)

    if result.success:
        print(f"✅ Berhasil!")
        print(f"Portal  : {result.source}")
        print(f"Judul   : {result.title}")
        print(f"Panjang : {len(result.content)} karakter")
        print(f"\nPreview konten (200 karakter pertama):")
        print(result.content[:200] + "...")
    else:
        print(f"❌ Gagal: {result.error_message}")
