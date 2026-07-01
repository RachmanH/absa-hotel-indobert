# absa-hotel-indobert

Aspect-Based Sentiment Analysis (ABSA) untuk ulasan hotel berbahasa Indonesia, menggunakan **IndoBERT-lite** fine-tuned pada dataset **HoASA (IndoNLU)**. Proyek UAS Natural Language Processing.

Youtube : https://www.youtube.com/watch?v=NJRxl7Y26a8&t=85s

Dataset : https://github.com/IndoNLP/indonlu/tree/master/dataset/hoasa_absa-airy

Model mengklasifikasikan sentimen (**positif / negatif / netral**) untuk 10 aspek layanan hotel sekaligus dari satu ulasan: `ac`, `air_panas`, `bau`, `general`, `kebersihan`, `linen`, `service`, `sunrise_meal`, `tv`, `wifi`.

## Struktur Proyek

```
.
├── app.py                                  # Aplikasi demo Streamlit
├── Aspect-Base-Sentiment-Analysis.ipynb    # Notebook riset: EDA, training, evaluasi
├── .streamlit/config.toml                  # Tema UI Streamlit
└── medium/                                 # Artefak model terlatih (gitignored, lihat di bawah)
```

## Model & Hasil

- **Backbone:** `indobenchmark/indobert-lite-base-p1` (~11.7M parameter)
- **Dataset:** HoASA (Hotel Aspect Sentiment Analysis), format wide → long (2.283 ulasan → 22.830 sampel ulasan-aspek)
- **Format input:** `[CLS] aspek [SEP] ulasan [SEP]` (sentence-pair classification)
- **Penanganan imbalance:** weighted cross-entropy (kelas `neut` mendominasi ~80% data)
- **Hasil test set:** F1-Macro = **0.8652** (target awal ≥ 0.70)

Detail lengkap (EDA, arsitektur, kurva training, analisis kesalahan per aspek) ada di `Aspect-Base-Sentiment-Analysis.ipynb`.

## Menjalankan Demo (Streamlit)

Model terlatih (`medium/`) tidak masuk git karena ukurannya besar (`model.safetensors` ~44MB). Untuk menjalankan `app.py`:

1. Reproduksi artefak model dengan menjalankan `Aspect-Base-Sentiment-Analysis.ipynb` sampai sel "Penyimpanan Model" — akan menghasilkan folder `medium/` berisi:
   - `model.safetensors`, `config.json` — backbone IndoBERT-lite
   - `tokenizer.json`, `tokenizer_config.json` — tokenizer
   - `head_and_config.pt` — bobot classifier head + metadata

2. Install dependencies:
   ```bash
   pip install streamlit torch pandas transformers
   ```

3. Jalankan app:
   ```bash
   streamlit run app.py
   ```

## Cara Kerja

Setiap ulasan dipasangkan dengan masing-masing dari 10 aspek (frasa Indonesia, mis. `ac` → "pendingin ruangan") lalu diklasifikasikan secara terpisah oleh model, menghasilkan sentimen + confidence per aspek. UI menampilkan ringkasan (jumlah positif/negatif/netral), detail per aspek (kartu & tabel), serta highlight aspek yang menonjol (non-netral).
