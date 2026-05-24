# 🗞️ Indonesian News Summarizer + NER
**COMP6885001 — Natural Language Processing | Final Project 2025/2026**

Aplikasi web untuk meringkas artikel berita Indonesia dan mengidentifikasi
entitas penting menggunakan IndoBART-v2 dan IndoBERT-NER.

---

## 📁 Struktur Project

```
news_summarizer/
│
├── app.py                  ← Main Streamlit app (jalankan ini)
├── requirements.txt        ← Semua dependency
│
├── scraper/
│   ├── __init__.py
│   └── news_scraper.py     ← Web scraper (detik.com, kompas.com, dll.)
│
├── models/
│   ├── __init__.py
│   ├── summarizer.py       ← IndoBART-v2 summarization
│   └── ner.py              ← IndoBERT NER extraction
│
└── utils/                  ← (akan diisi: ROUGE evaluator, fine-tuning script)
```

---

## 🚀 Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan aplikasi
```bash
streamlit run app.py
```

Buka browser ke `http://localhost:8501`

---

## 🧩 Komponen NLP

| Komponen | Model | Dataset Training |
|---|---|---|
| Summarization | `indobenchmark/indobart-v2` | Liputan6 (215K pasang) |
| NER | `cahya/bert-base-indonesian-NER` | IndoNLU/NERGrit |
| Scraper | BeautifulSoup | - |

### Label NER
| Label | Keterangan | Contoh |
|---|---|---|
| PER | Nama tokoh | Prabowo Subianto |
| ORG | Organisasi | Bank Indonesia |
| LOC | Lokasi | Selat Malaka |
| GPE | Entitas geopolitik | Indonesia |
| DATE | Waktu/tanggal | 20 Februari 2026 |
| EVT | Peristiwa | Pemilu 2024 |

---

## 📊 Evaluasi

- **Summarization**: ROUGE-1, ROUGE-2, ROUGE-L (test set IndoSum)
- **NER**: Precision, Recall, F1-Score per label

---

## 👥 Tim
- [Nama Anggota 1]
- [Nama Anggota 2]
- [Nama Anggota 3]
