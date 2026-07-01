import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from transformers import BertTokenizerFast, AutoModel

MODEL_DIR = Path(__file__).parent / "medium"

ASPECT_DISPLAY = {
    "ac":           "pendingin ruangan",
    "air_panas":    "air panas",
    "bau":          "bau",
    "general":      "kesan umum",
    "kebersihan":   "kebersihan",
    "linen":        "linen",
    "service":      "pelayanan",
    "sunrise_meal": "sarapan pagi",
    "tv":           "televisi",
    "wifi":         "wifi",
}

SENTIMENT_COLOR = {"pos": "#dcfce7", "neg": "#fee2e2", "neut": "#f1f5f9"}
SENTIMENT_LABEL = {"pos": "Positif ✅", "neg": "Negatif ❌", "neut": "Netral ➖"}

DEMO_REVIEWS = [
    "Kamar sangat bersih dan AC-nya dingin sekali, tapi wifi lambat banget.",
    "Pelayanan staf sangat ramah, sarapan enak, tapi kamar bau dan TV rusak.",
    "Hotel biasa saja, tidak ada yang istimewa maupun mengecewakan.",
]


class IndoBERTForABSA(nn.Module):
    def __init__(self, backbone, num_classes=3):
        super().__init__()
        self.bert = backbone
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(backbone.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.fc(self.dropout(cls))


@st.cache_resource(show_spinner="Memuat model IndoBERT...")
def load_model():
    ckpt = torch.load(MODEL_DIR / "head_and_config.pt", map_location="cpu")
    tokenizer = BertTokenizerFast.from_pretrained(str(MODEL_DIR))
    backbone = AutoModel.from_pretrained(str(MODEL_DIR))
    model = IndoBERTForABSA(backbone, ckpt["num_classes"])
    model.fc.load_state_dict(ckpt["fc_state_dict"])
    model.eval()
    return (
        model,
        tokenizer,
        ckpt["aspects"],
        ckpt["target_names"],
        ckpt["config"]["max_len"],
    )


def predict(text, model, tokenizer, aspects, target_names, max_len):
    results = []
    with torch.no_grad():
        for asp in aspects:
            aspect_text = ASPECT_DISPLAY.get(asp, asp)
            enc = tokenizer(
                aspect_text, text,
                max_length=max_len,
                padding="max_length",
                truncation="only_second",
                return_tensors="pt",
            )
            logits = model(enc["input_ids"], enc["attention_mask"])
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred = int(probs.argmax())
            results.append({
                "aspect_key": asp,
                "Aspek": ASPECT_DISPLAY.get(asp, asp).title(),
                "Sentimen": target_names[pred],
                "Confidence": float(probs[pred]),
            })
    return pd.DataFrame(results)


ASPECT_ICON = {
    "ac": "❄️", "air_panas": "🚿", "bau": "👃", "general": "🏨",
    "kebersihan": "🧹", "linen": "🛏️", "service": "🛎️",
    "sunrise_meal": "🍳", "tv": "📺", "wifi": "📶",
}

# ── UI ────────────────────────────────────────────────────
st.set_page_config(page_title="ABSA Hotel Indonesia", page_icon="🏨", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 900px; }

    h1 { font-weight: 700; color: #1e293b; letter-spacing: -0.02em; }
    h2, h3 { font-weight: 600; color: #334155; }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 12px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stMetricLabel"] { color: #64748b; }

    div[data-testid="stTextArea"] textarea {
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        background: #f8fafc;
    }

    div[data-testid="stContainer"] {
        border-radius: 12px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }

    button[kind="primary"] {
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);
    }

    section[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }

    .aspect-card {
        border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 10px 14px; margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🏨 ABSA Hotel")
    st.markdown(
        "Aspect-Based Sentiment Analysis ulasan hotel berbahasa Indonesia, "
        "10 aspek sekaligus dari satu ulasan."
    )
    st.markdown("**Model:** IndoBERT-lite\n\n**Dataset:** HoASA (IndoNLU)")
    st.divider()
    st.caption("Sentimen: ✅ Positif · ❌ Negatif · ➖ Netral")

st.title("🏨 ABSA Hotel Indonesia")
st.caption(
    "Aspect-Based Sentiment Analysis ulasan hotel — "
    "IndoBERT-lite fine-tuned pada dataset HoASA (IndoNLU)"
)

model, tokenizer, aspects, target_names, max_len = load_model()

with st.expander("💡 Contoh review (klik untuk menyalin)"):
    for r in DEMO_REVIEWS:
        st.code(r, language=None)

text = st.text_area(
    "Masukkan ulasan hotel (Bahasa Indonesia):",
    placeholder="Contoh: Kamar sangat bersih dan AC-nya dingin, tapi wifi lambat banget.",
    height=120,
)

col_btn, col_count = st.columns([1, 3])
with col_btn:
    analyze = st.button("Analisis Sentimen", type="primary", disabled=not text.strip(), use_container_width=True)
with col_count:
    if text.strip():
        st.caption(f"{len(text)} karakter")

if analyze and text.strip():
    with st.spinner("Menganalisis 10 aspek..."):
        df = predict(text, model, tokenizer, aspects, target_names, max_len)

    st.divider()
    st.success("Analisis selesai ✅")

    # ── Ringkasan metrik ──────────────────────────────────
    pos_n = (df["Sentimen"] == "pos").sum()
    neg_n = (df["Sentimen"] == "neg").sum()
    neut_n = (df["Sentimen"] == "neut").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Positif ✅", pos_n)
    c2.metric("Negatif ❌", neg_n)
    c3.metric("Netral ➖", neut_n)

    st.divider()

    # ── Detail per aspek (card + progress bar) ────────────
    st.subheader("Detail Sentimen per Aspek")
    tab_cards, tab_table = st.tabs(["🗂️ Kartu", "📋 Tabel"])

    with tab_cards:
        cols = st.columns(2)
        for i, r in enumerate(df.itertuples()):
            with cols[i % 2]:
                icon = ASPECT_ICON.get(r.aspect_key, "•")
                with st.container(border=True):
                    st.markdown(f"**{icon} {r.Aspek}** — {SENTIMENT_LABEL[r.Sentimen]}")
                    st.progress(r.Confidence, text=f"{r.Confidence*100:.1f}% confidence")

    with tab_table:
        display_df = df[["Aspek", "Sentimen", "Confidence"]].copy()
        display_df["Sentimen"] = display_df["Sentimen"].map(SENTIMENT_LABEL)
        display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x*100:.1f}%")

        def row_color(row):
            key = [k for k, v in SENTIMENT_LABEL.items() if v == row["Sentimen"]]
            color = SENTIMENT_COLOR.get(key[0] if key else "neut", "#f5f5f5")
            return [f"color: black; background-color: {color}"] * len(row)

        st.dataframe(
            display_df.style.apply(row_color, axis=1),
            use_container_width=True,
            hide_index=True,
        )

    # ── Highlight non-netral ──────────────────────────────
    non_neut = df[df["Sentimen"] != "neut"]
    if not non_neut.empty:
        st.info(
            f"**Aspek menonjol:** "
            + ", ".join(
                f"{ASPECT_DISPLAY.get(r.aspect_key, r.aspect_key)} ({r.Sentimen})"
                for r in non_neut.itertuples()
            )
        )
    else:
        st.info("Semua aspek: netral — tidak ada sentimen yang menonjol.")
