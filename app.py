"""
app.py
======
Main Streamlit application untuk Indonesian News Summarizer + NER.
Jalankan dengan: streamlit run app.py
"""

import streamlit as st
import sys
import os

# Tambah root project ke path
sys.path.insert(0, os.path.dirname(__file__))

from scraper.news_scraper import scrape_article
from models.summarizer import Summarizer
from models.ner import NERExtractor, LABEL_CONFIG

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BeritaRingkas — Indonesian News Summarizer",
    page_icon="🗞️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — mirip referensi: gradient background, card putih, clean typography
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ─── Global ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #EEF2FF 0%, #F5F0FF 50%, #EFF6FF 100%);
    min-height: 100vh;
}

/* Sembunyikan elemen Streamlit default yang tidak perlu */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ─── Hero Section ───────────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.1);
    color: #6366F1;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    border: 1px solid rgba(99, 102, 241, 0.2);
    margin-bottom: 1.2rem;
    text-transform: uppercase;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #1E1B4B;
    line-height: 1.2;
    margin-bottom: 1rem;
}
.hero-title span {
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #6B7280;
    max-width: 520px;
    margin: 0 auto 2rem;
    line-height: 1.6;
}

/* ─── Input Card ─────────────────────────────────────────────────────── */
.input-card {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.08),
                0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99, 102, 241, 0.08);
}

/* ─── Streamlit Input Override ───────────────────────────────────────── */
.stTextInput > div > div > input {
    border-radius: 12px !important;
    border: 2px solid #E5E7EB !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s !important;
    background: #FAFAFA !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366F1 !important;
    background: white !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* ─── Button ─────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45) !important;
}

/* ─── Result Card ────────────────────────────────────────────────────── */
.result-card {
    background: white;
    border-radius: 20px;
    padding: 1.8rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
    border: 1px solid #F3F4F6;
}
.result-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
    padding-bottom: 0.8rem;
    border-bottom: 2px solid #F3F4F6;
}
.result-header-icon {
    font-size: 1.3rem;
}
.result-header-title {
    font-size: 1rem;
    font-weight: 700;
    color: #1E1B4B;
}
.result-header-badge {
    margin-left: auto;
    background: #EEF2FF;
    color: #6366F1;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
}
.summary-text {
    font-size: 1rem;
    color: #374151;
    line-height: 1.75;
    background: #FAFBFF;
    border-left: 4px solid #6366F1;
    padding: 1rem 1.2rem;
    border-radius: 0 12px 12px 0;
}

/* ─── NER Tags ───────────────────────────────────────────────────────── */
.ner-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.ner-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 500;
    border: 1.5px solid;
}
.ner-category {
    margin-bottom: 1rem;
}
.ner-category-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #9CA3AF;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

/* ─── Article Meta ───────────────────────────────────────────────────── */
.article-meta {
    background: #F9FAFB;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    border: 1px solid #F3F4F6;
}
.meta-title {
    font-weight: 600;
    color: #1F2937;
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
}
.meta-source {
    font-size: 0.8rem;
    color: #9CA3AF;
}

/* ─── Stats Row ──────────────────────────────────────────────────────── */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}
.stat-box {
    flex: 1;
    background: #F9FAFB;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    text-align: center;
    border: 1px solid #F3F4F6;
}
.stat-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #6366F1;
}
.stat-label {
    font-size: 0.75rem;
    color: #9CA3AF;
    margin-top: 0.1rem;
}

/* ─── Error / Info ───────────────────────────────────────────────────── */
.error-box {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #DC2626;
    font-size: 0.9rem;
}
.info-box {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    color: #1D4ED8;
    font-size: 0.85rem;
    text-align: center;
}

/* ─── Footer ─────────────────────────────────────────────────────────── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #D1D5DB;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — supaya model tidak reload setiap kali ada interaksi
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    """Load model sekali, cache di memori."""
    summarizer = Summarizer()
    summarizer.load()
    ner = NERExtractor()
    ner.load()
    return summarizer, ner


# ══════════════════════════════════════════════════════════════════════════════
# HELPER RENDER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def render_ner_results(ner_result):
    """Render hasil NER sebagai tag berwarna per kategori."""
    grouped = ner_result.unique_by_label()
    if not grouped:
        st.markdown('<p style="color:#9CA3AF;font-size:0.9rem;">Tidak ada entitas terdeteksi.</p>',
                    unsafe_allow_html=True)
        return

    for label, texts in grouped.items():
        config = LABEL_CONFIG[label]
        color = config["color"]
        bg = color + "15"   # warna transparan 15%

        tags_html = "".join([
            f'<span class="ner-tag" style="background:{bg};color:{color};border-color:{color}40;">'
            f'{config["emoji"]} {text}</span>'
            for text in texts
        ])

        st.markdown(f"""
        <div class="ner-category">
            <div class="ner-category-label">{config['emoji']} {config['label']}</div>
            <div class="ner-grid">{tags_html}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🇮🇩 NLP · Bahasa Indonesia</div>
    <h1 class="hero-title">Ringkas Berita dengan<br><span>Satu Klik</span></h1>
    <p class="hero-subtitle">
        Paste URL artikel berita Indonesia — sistem akan otomatis menghasilkan
        ringkasan dan mengidentifikasi tokoh, lokasi, organisasi, dan peristiwa penting.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Input Card ─────────────────────────────────────────────────────────────────
st.markdown('<div class="input-card">', unsafe_allow_html=True)

url_input = st.text_input(
    label="URL Artikel",
    placeholder="https://www.detik.com/...",
    label_visibility="collapsed"
)

col1, col2 = st.columns([2, 1])
with col1:
    length_choice = st.selectbox(
        "Panjang Ringkasan",
        options=["pendek", "sedang", "panjang"],
        index=1,
        format_func=lambda x: {
            "pendek": "📄 Pendek (1–2 kalimat)",
            "sedang": "📃 Sedang (3–4 kalimat)",
            "panjang": "📋 Panjang (5+ kalimat)"
        }[x],
        label_visibility="collapsed"
    )
with col2:
    process_btn = st.button("✨ Proses Artikel", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Contoh URL ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="info-box">
    💡 Mendukung: detik.com · kompas.com · tribunnews.com · tempo.co · dan lainnya
</div>
""", unsafe_allow_html=True)

# ── Proses ─────────────────────────────────────────────────────────────────────
if process_btn:
    if not url_input.strip():
        st.markdown('<div class="error-box">⚠️ Masukkan URL artikel terlebih dahulu.</div>',
                    unsafe_allow_html=True)
    elif not url_input.startswith("http"):
        st.markdown('<div class="error-box">⚠️ URL tidak valid. Pastikan dimulai dengan https://</div>',
                    unsafe_allow_html=True)
    else:
        # Step 1: Scraping
        with st.spinner("🔍 Mengambil artikel..."):
            article = scrape_article(url_input)

        if not article.success:
            st.markdown(f'<div class="error-box">❌ Gagal mengambil artikel: {article.error_message}</div>',
                        unsafe_allow_html=True)
        else:
            # Tampilkan meta artikel
            st.markdown(f"""
            <div class="article-meta">
                <div class="meta-title">📰 {article.title}</div>
                <div class="meta-source">🌐 {article.source} · {len(article.content.split())} kata</div>
            </div>
            """, unsafe_allow_html=True)

            # Load model (sudah di-cache)
            with st.spinner("⚙️ Memuat model NLP..."):
                summarizer, ner_extractor = load_models()

            # Step 2: Summarization
            with st.spinner("📝 Membuat ringkasan..."):
                summary_result = summarizer.summarize(article.content, length=length_choice)

            # Step 3: NER
            with st.spinner("🏷️ Mengidentifikasi entitas..."):
                ner_result = ner_extractor.extract(article.content)

            # ── Tampilkan hasil ─────────────────────────────────────────────

            # Ringkasan
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("""
            <div class="result-header">
                <span class="result-header-icon">📝</span>
                <span class="result-header-title">Ringkasan Artikel</span>
                <span class="result-header-badge">IndoBART-v2</span>
            </div>
            """, unsafe_allow_html=True)

            if summary_result.success:
                st.markdown(f'<div class="summary-text">{summary_result.summary}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="error-box">❌ {summary_result.error_message}</div>',
                            unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # NER
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-header">
                <span class="result-header-icon">🏷️</span>
                <span class="result-header-title">Entitas Terdeteksi</span>
                <span class="result-header-badge">{ner_result.total} entitas</span>
            </div>
            """, unsafe_allow_html=True)

            if ner_result.success:
                render_ner_results(ner_result)
            else:
                st.markdown(f'<div class="error-box">❌ {ner_result.error_message}</div>',
                            unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Stats
            st.markdown(f"""
            <div class="stats-row">
                <div class="stat-box">
                    <div class="stat-value">{len(article.content.split())}</div>
                    <div class="stat-label">Kata Artikel</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len(summary_result.summary.split()) if summary_result.success else 0}</div>
                    <div class="stat-label">Kata Ringkasan</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{ner_result.total}</div>
                    <div class="stat-label">Entitas</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">
                        {round((1 - len(summary_result.summary.split()) / max(len(article.content.split()), 1)) * 100) if summary_result.success else 0}%
                    </div>
                    <div class="stat-label">Kompresi</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    COMP6885001 · Natural Language Processing · Final Project 2025/2026<br>
    Powered by IndoBART-v2 + IndoBERT-NER · Binus University
</div>
""", unsafe_allow_html=True)
