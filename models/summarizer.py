"""
models/summarizer.py
====================
Komponen summarization menggunakan bert2gpt-indonesian-summarization.
Menggunakan model pre-trained tanpa fine-tuning tambahan, berdasarkan
hasil evaluasi ROUGE yang menunjukkan performa terbaik dibanding
varian yang di-fine-tune pada IndoSum maupun Liputan6.
"""

import re
import torch
from transformers import BertTokenizer, EncoderDecoderModel
from dataclasses import dataclass
from typing import Optional


# ── Konstanta ──────────────────────────────────────────────────────────────────
BASE_MODEL       = "cahya/bert2gpt-indonesian-summarization"
MAX_INPUT_LENGTH = 512


@dataclass
class SummaryResult:
    summary: str
    input_length: int
    output_length: int
    success: bool
    error_message: Optional[str] = None


SUMMARY_LENGTH = {
    "pendek":  {"min": 15, "max": 35},
    "sedang":  {"min": 40, "max": 70},
    "panjang": {"min": 70, "max": 110},
}

# Kata-kata umum Indonesia yang TIDAK boleh di-capitalize otomatis
COMMON_WORDS = {
    "program", "beasiswa", "pendidikan", "pemerintah", "menteri",
    "dalam", "untuk", "yang", "dari", "dengan", "pada", "oleh",
    "ini", "itu", "akan", "sudah", "telah", "juga", "bagi",
    "atau", "dan", "ke", "di", "se", "per", "atas", "bawah",
    "portal", "resmi", "nasional", "tinggi", "penuh", "mulai",
    "bulan", "depan", "melalui", "seluruh", "daerah", "akses",
    "saat", "ketika", "bahwa", "karena", "sehingga", "namun",
    "tetapi", "namun", "meski", "walaupun", "agar", "supaya",
    "adapun", "sedangkan", "kemudian", "lalu", "setelah", "sebelum",
    "antara", "hingga", "sampai", "sejak", "sekitar", "sekitar",
    "baru", "lama", "besar", "kecil", "tinggi", "rendah", "panjang",
    "juga", "hanya", "bahkan", "justru", "malah", "sudah", "belum",
    "masih", "sudah", "telah", "akan", "ingin", "mau", "bisa",
    "dapat", "harus", "perlu", "boleh", "tidak", "bukan", "jangan",
    "rencana", "upaya", "langkah", "kebijakan", "kegiatan", "acara",
    "hasil", "proses", "sistem", "bidang", "sektor", "wilayah",
    "jumlah", "angka", "data", "informasi", "laporan", "dokumen",
}

# Kata pendek umum yang tidak dihapus saat filter garbage
COMMON_SHORT = {
    "di", "ke", "dan", "atau", "yang", "ini", "itu", "ia", "si",
    "pun", "tak", "rp", "ri", "as", "ui", "km", "kg", "ha",
}


# ── Post-processing helpers ────────────────────────────────────────────────────
def _build_case_map(original_text: str) -> dict:
    """
    Bangun mapping lowercase ke proper case dari teks artikel asli.
    Hanya untuk kata yang bukan kata umum (proper nouns: nama orang, kota, negara, dll).
    """
    case_map = {}
    for word in original_text.split():
        clean = re.sub(r'^[^\w]+|[^\w]+$', '', word)
        if not clean or len(clean) <= 1:
            continue
        lower = clean.lower()
        if lower in COMMON_WORDS:
            continue
        if lower not in case_map or clean[0].isupper():
            case_map[lower] = clean
    return case_map


def _apply_case_map(text: str, case_map: dict) -> str:
    """Terapkan kapitalisasi proper noun dari artikel asli ke ringkasan."""
    result = []
    for token in text.split():
        prefix = re.match(r'^([^\w]*)', token).group(1)
        suffix = re.search(r'([^\w]*)$', token).group(1)
        core   = token[len(prefix):len(token) - len(suffix)] if suffix else token[len(prefix):]
        if core:
            proper = case_map.get(core.lower(), core)
            result.append(prefix + proper + suffix)
        else:
            result.append(token)
    return ' '.join(result)


def _remove_garbage_words(text: str, case_map: dict) -> str:
    """Hapus kata-kata garbage hasil hallucination model."""
    words   = text.split()
    cleaned = []
    for w in words:
        core = re.sub(r'^[^\w]+|[^\w]+$', '', w)
        if re.match(r'^[\d.,]+$', core):
            cleaned.append(w)
            continue
        if len(core) > 20:
            continue
        if len(core) <= 2 and core.lower() not in case_map and core.lower() not in COMMON_SHORT:
            continue
        cleaned.append(w)
    return ' '.join(cleaned)


def _postprocess(text: str, case_map: dict = None) -> str:
    """
    Bersihkan dan perbaiki teks ringkasan:
    1. Hapus kata garbage
    2. Fix spasi tanda baca dan apostrof
    3. Fix angka (50. 000 -> 50.000)
    4. Pecah per kalimat, filter kalimat pendek dan duplikat
    5. Apply case map (proper nouns dari artikel asli)
    6. Capitalize huruf pertama tiap kalimat
    """
    if not text:
        return text

    if case_map:
        text = _remove_garbage_words(text, case_map)

    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)

    text = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', text)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned   = []
    seen_words_sets = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s.split()) < 5:
            continue
        words_set = set(s.lower().split())
        is_dup = False
        for prev_set in seen_words_sets:
            if prev_set and len(words_set & prev_set) / max(len(words_set), 1) > 0.6:
                is_dup = True
                break
        if is_dup:
            continue
        seen_words_sets.append(words_set)
        if case_map:
            s = _apply_case_map(s, case_map)
        s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
        if not s.endswith(('.', '!', '?')):
            s += '.'
        cleaned.append(s)
    if not cleaned:
        return text[0].upper() + text[1:] if text else text
    return ' '.join(cleaned)


# ── Class utama ────────────────────────────────────────────────────────────────
class Summarizer:
    """
    Wrapper untuk model bert2gpt Indonesian summarization.
    Menggunakan checkpoint pre-trained cahya/bert2gpt-indonesian-summarization
    tanpa fine-tuning tambahan, sesuai hasil evaluasi ROUGE pada Liputan6.
    """

    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or BASE_MODEL
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model      = None
        self.tokenizer  = None
        self._loaded    = False
        print(f"[Summarizer] Model: {self.model_path}")
        print(f"[Summarizer] Device: {self.device}")

    def load(self):
        if self._loaded:
            return
        print(f"[Summarizer] Loading model '{self.model_path}'...")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.eos_token = self.tokenizer.sep_token
        self.model = EncoderDecoderModel.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        print("[Summarizer] Model berhasil dimuat.")

    def summarize(self, text: str, length: str = "sedang") -> SummaryResult:
        """
        Generate ringkasan dari teks input.

        Args:
            text   : Teks artikel yang ingin diringkas
            length : "pendek", "sedang", atau "panjang"

        Returns:
            SummaryResult berisi ringkasan dan metadata.
        """
        if not self._loaded:
            self.load()

        if not text or len(text.strip()) < 50:
            return SummaryResult(
                summary="", input_length=0, output_length=0,
                success=False,
                error_message="Teks terlalu pendek untuk diringkas."
            )

        length_config = SUMMARY_LENGTH.get(length, SUMMARY_LENGTH["sedang"])

        try:
            case_map = _build_case_map(text)

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
                padding="max_length",
            ).to(self.device)

            input_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    min_length=length_config["min"],
                    max_length=length_config["max"],
                    num_beams=3,
                    repetition_penalty=1.8,
                    length_penalty=1.0,
                    early_stopping=False,
                    no_repeat_ngram_size=4,
                    use_cache=True,
                    do_sample=False,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            raw_summary = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            summary = _postprocess(raw_summary, case_map)

            return SummaryResult(
                summary=summary,
                input_length=input_length,
                output_length=len(output_ids[0]),
                success=True
            )

        except Exception as e:
            return SummaryResult(
                summary="", input_length=0, output_length=0,
                success=False,
                error_message=f"Error saat generate ringkasan: {str(e)}"
            )

    def is_loaded(self) -> bool:
        return self._loaded


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_text = """
    Jakarta - Pemerintah Indonesia resmi meluncurkan program beasiswa nasional
    untuk mendukung pendidikan tinggi bagi generasi muda Indonesia. Program ini
    diumumkan oleh Menteri Pendidikan dalam acara yang dihadiri oleh ribuan
    mahasiswa di Jakarta. Beasiswa ini mencakup biaya kuliah penuh, tunjangan
    hidup, dan biaya penelitian. Program ini ditargetkan untuk 50.000 mahasiswa
    berprestasi dari seluruh Indonesia, khususnya dari daerah terpencil yang
    memiliki keterbatasan akses terhadap pendidikan berkualitas. Pendaftaran
    dibuka mulai bulan depan melalui portal resmi Kemendikbud.
    """

    summarizer = Summarizer()
    result = summarizer.summarize(sample_text, length="sedang")

    if result.success:
        print(f"Ringkasan:\n{result.summary}")
        print(f"\nInput tokens : {result.input_length}")
        print(f"Output tokens: {result.output_length}")
    else:
        print(f"Error: {result.error_message}")