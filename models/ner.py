"""
models/ner.py
=============
Komponen Named Entity Recognition (NER) menggunakan
IndoBERT yang sudah di-pretrain untuk NER Bahasa Indonesia.
"""

import re
from transformers import pipeline
from dataclasses import dataclass, field
from typing import List, Optional


# ── Konstanta ──────────────────────────────────────────────────────────────────
MODEL_NAME = "cahya/bert-base-indonesian-NER"

# Mapping label → emoji + warna (untuk UI)
LABEL_CONFIG = {
    "PER":  {"label": "Tokoh",      "emoji": "👤", "color": "#4A90D9"},
    "ORG":  {"label": "Organisasi", "emoji": "🏛️", "color": "#E67E22"},
    "LOC":  {"label": "Lokasi",     "emoji": "📍", "color": "#27AE60"},
    "GPE":  {"label": "Geopolitik", "emoji": "🌏", "color": "#8E44AD"},
    "DATE": {"label": "Waktu",      "emoji": "📅", "color": "#E74C3C"},
    "EVT":  {"label": "Peristiwa",  "emoji": "⚡", "color": "#F39C12"},
}

# Mapping label alternatif → label kita
LABEL_MAP = {
    "PERSON":       "PER",
    "PEOPLE":       "PER",
    "NAMA":         "PER",
    "ORGANIZATION": "ORG",
    "ORGANISATION": "ORG",
    "COMPANY":      "ORG",
    "LEMBAGA":      "ORG",
    "LOCATION":     "LOC",
    "PLACE":        "LOC",
    "LOKASI":       "LOC",
    "COUNTRY":      "GPE",
    "GEOPOLITICAL": "GPE",
    "NATION":       "GPE",
    "TIME":         "DATE",
    "DATETIME":     "DATE",
    "WAKTU":        "DATE",
    "TANGGAL":      "DATE",
    "EVENT":        "EVT",
    "KEJADIAN":     "EVT",
}

# Kata-kata umum yang sering salah terdeteksi sebagai entitas
STOPWORDS = {
    "with", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "atau", "yang", "dari", "dan", "ke", "di", "ini", "itu", "ada",
    "oleh", "para", "juga", "bisa", "akan", "sudah", "saat", "agar",
    "jika", "saja", "pun", "bagi", "atas", "bawah", "dalam", "luar",
    "rdp", "rp", "jp", "ar", "ab", "tp", "dr", "mr", "ms",
}


# ── Data class ─────────────────────────────────────────────────────────────────
@dataclass
class Entity:
    """Satu entitas yang ditemukan dalam teks."""
    text: str
    label: str
    score: float
    start: int
    end: int

    @property
    def emoji(self) -> str:
        return LABEL_CONFIG.get(self.label, {}).get("emoji", "🏷️")

    @property
    def label_display(self) -> str:
        return LABEL_CONFIG.get(self.label, {}).get("label", self.label)

    @property
    def color(self) -> str:
        return LABEL_CONFIG.get(self.label, {}).get("color", "#999999")


@dataclass
class NERResult:
    """Menyimpan semua entitas yang ditemukan dari satu teks."""
    entities: List[Entity] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None

    def by_label(self, label: str) -> List[Entity]:
        return [e for e in self.entities if e.label == label]

    def unique_by_label(self) -> dict:
        result = {}
        for label in LABEL_CONFIG.keys():
            entities = self.by_label(label)
            unique_texts = list(dict.fromkeys(e.text for e in entities))
            if unique_texts:
                result[label] = unique_texts
        return result

    @property
    def total(self) -> int:
        return len(self.entities)


# ── Class utama ────────────────────────────────────────────────────────────────
class NERExtractor:
    """
    Wrapper untuk model NER Bahasa Indonesia.

    Cara pakai:
        ner = NERExtractor()
        result = ner.extract(text)
        for entity in result.entities:
            print(f"{entity.emoji} {entity.text} → {entity.label}")
    """

    def __init__(self, model_path: str = MODEL_NAME):
        self.model_path = model_path
        self._pipeline  = None
        self._loaded    = False

    def load(self):
        """Load model NER ke memori."""
        if self._loaded:
            return
        print(f"[NER] Loading model '{self.model_path}'...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, model_max_length=512)
        self._pipeline = pipeline(
            task="ner",
            model=self.model_path,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )
        self._loaded = True
        print("[NER] Model berhasil dimuat!")

    def _clean_text(self, text: str) -> str:
        """Bersihkan teks dari karakter aneh."""
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _is_valid_entity(self, text: str) -> bool:
        """Filter entitas yang tidak valid."""
        text = text.strip()
        # Minimal 3 karakter
        if len(text) < 3:
            return False
        # Jangan hanya angka atau simbol
        if re.match(r'^[\d\s\.\,\-\_]+$', text):
            return False
        # Jangan hanya satu kata dengan panjang <= 2
        words = text.split()
        if len(words) == 1 and len(text) <= 2:
            return False
        # Filter stopwords dan singkatan umum
        if text.lower() in STOPWORDS:
            return False
        # Filter sisa tokenisasi BERT (## prefix)
        if text.startswith("##"):
            return False
        # Filter teks yang mengandung tanda kurung saja
        if re.match(r'^[\(\)\[\]\{\}]+$', text):
            return False
        return True

    def _fix_text(self, text: str) -> str:
        """Perbaiki teks entitas hasil tokenisasi."""
        # Fix spasi di sekitar apostrof
        text = re.sub(r"\s+'", "'", text)
        text = re.sub(r"'\s+", "'", text)
        # Fix spasi sebelum tanda baca
        text = re.sub(r'\s+([.,;:])', r'\1', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract(self, text: str, min_score: float = 0.65) -> NERResult:
        """
        Ekstrak entitas dari teks.

        Args:
            text      : Teks artikel
            min_score : Threshold confidence minimum (default 0.65)

        Returns:
            NERResult berisi list Entity yang ditemukan.
        """
        if not self._loaded:
            self.load()

        if not text or len(text.strip()) < 10:
            return NERResult(
                success=False,
                error_message="Teks terlalu pendek untuk NER."
            )

        try:
            # Bagi teks jadi chunk 400 kata supaya tidak melebihi 512 token BERT
            words  = text.split()
            chunks = [" ".join(words[i:i+250]) for i in range(0, len(words), 250)]

            raw_results = []
            for chunk in chunks:
                chunk_clean = self._clean_text(chunk)
                if chunk_clean:
                    raw_results.extend(self._pipeline(chunk_clean))

            entities = []
            for item in raw_results:
                # Filter score
                if item["score"] < min_score:
                    continue

                # Bersihkan dan map label
                raw_label = item["entity_group"].replace("B-", "").replace("I-", "").upper()
                label     = LABEL_MAP.get(raw_label, raw_label)

                # Hanya ambil label yang kita support
                if label not in LABEL_CONFIG:
                    continue

                # Bersihkan teks entitas
                entity_text = self._fix_text(item["word"])

                # Filter entitas tidak valid
                if not self._is_valid_entity(entity_text):
                    continue

                entity = Entity(
                    text=entity_text,
                    label=label,
                    score=round(item["score"], 3),
                    start=item.get("start", 0),
                    end=item.get("end", 0),
                )
                entities.append(entity)

            # Hapus duplikat (case-insensitive)
            seen = set()
            unique_entities = []
            for e in entities:
                key = (e.text.lower(), e.label)
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(e)

            # Urutkan berdasarkan label
            label_order = list(LABEL_CONFIG.keys())
            unique_entities.sort(
                key=lambda e: label_order.index(e.label) if e.label in label_order else 99
            )

            return NERResult(entities=unique_entities, success=True)

        except Exception as ex:
            return NERResult(
                success=False,
                error_message=f"Error saat ekstraksi NER: {str(ex)}"
            )

    def is_loaded(self) -> bool:
        return self._loaded


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_text = """
    Presiden Prabowo Subianto bertemu dengan Perdana Menteri Malaysia Anwar Ibrahim
    di Kuala Lumpur pada Senin, 20 Februari 2026. Pertemuan ini membahas kerja sama
    bilateral antara Indonesia dan Malaysia di bidang ekonomi. Turut hadir Menteri
    Luar Negeri Sugiono dan perwakilan dari Bank Indonesia serta Kementerian Keuangan.
    Gubernur Kalimantan Timur Rudy Mas'ud juga menyatakan dukungannya terhadap kebijakan ini.
    """

    ner = NERExtractor()
    result = ner.extract(sample_text)

    if result.success:
        print(f"✅ Ditemukan {result.total} entitas:\n")
        for entity in result.entities:
            print(f"  {entity.emoji} [{entity.label}] {entity.text} (score: {entity.score})")

        print("\n📊 Ringkasan per kategori:")
        for label, texts in result.unique_by_label().items():
            config = LABEL_CONFIG[label]
            print(f"  {config['emoji']} {config['label']}: {', '.join(texts)}")
    else:
        print(f"❌ Error: {result.error_message}")
