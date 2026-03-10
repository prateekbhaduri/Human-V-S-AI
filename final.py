"""
AI vs Human Text Classifier — Interactive Streamlit Dashboard
Dataset: ai_vs_human_dataset_medium.csv
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="AI vs Human Classifier", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════════════════
#  STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
.stApp{background:#0c0e12;color:#d4d8e2;}
[data-testid="stSidebar"]{background:#0a0c10;border-right:1px solid #1e2330;}
[data-testid="stSidebar"] *{color:#b0b8cc;}

.hero{background:linear-gradient(135deg,#0f1420 0%,#111827 60%,#0a0f1a 100%);
  border:1px solid #1e2330;border-radius:14px;padding:2.4rem 3rem 2rem;margin-bottom:1.6rem;
  position:relative;overflow:hidden;}
.hero::before{content:"";position:absolute;top:-80px;right:-80px;width:280px;height:280px;
  border-radius:50%;background:radial-gradient(circle,rgba(245,166,35,.1) 0%,transparent 65%);}
.hero-eyebrow{font-family:'IBM Plex Mono',monospace;font-size:.68rem;letter-spacing:.18em;
  text-transform:uppercase;color:#f5a623;margin-bottom:.5rem;}
.hero-title{font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:700;
  color:#eef0f4;margin:0 0 .6rem;line-height:1.2;}
.hero-title em{color:#f5a623;font-style:normal;}
.hero-desc{font-size:.9rem;color:#6b7591;line-height:1.7;max-width:640px;margin:0;}


.sec{font-family:'IBM Plex Mono',monospace;font-size:.68rem;letter-spacing:.14em;
  text-transform:uppercase;color:#6b7591;padding:.4rem 0;
  border-bottom:1px solid #1e2330;margin:1.8rem 0 1rem;}

.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.9rem;margin-bottom:1.2rem;}
.stat{background:#101520;border:1px solid #1e2330;border-radius:10px;
  padding:1.2rem 1.4rem;position:relative;overflow:hidden;}
.stat::after{content:"";position:absolute;bottom:0;left:0;right:0;height:2px;}
.stat.amber::after{background:#f5a623;} .stat.teal::after{background:#38c9b0;}
.stat.rose::after{background:#f06292;} .stat.sky::after{background:#56b4e9;}
.stat-label{font-family:'IBM Plex Mono',monospace;font-size:.62rem;letter-spacing:.1em;
  text-transform:uppercase;color:#6b7591;margin-bottom:.3rem;}
.stat-val{font-family:'IBM Plex Mono',monospace;font-size:1.9rem;font-weight:700;
  color:#eef0f4;line-height:1;}
.stat-sub{font-size:.72rem;color:#f5a623;margin-top:.25rem;}

.res-card{border-radius:10px;padding:1.3rem 1.5rem;text-align:center;margin:.3rem 0;}
.res-human{background:rgba(56,201,176,.08);border:1px solid rgba(56,201,176,.25);}
.res-ai{background:rgba(240,98,146,.08);border:1px solid rgba(240,98,146,.25);}
.res-model{font-family:'IBM Plex Mono',monospace;font-size:.6rem;letter-spacing:.12em;
  text-transform:uppercase;color:#6b7591;margin-bottom:.3rem;}
.res-verdict{font-size:1.25rem;font-weight:600;}
.col-human{color:#38c9b0;} .col-ai{color:#f06292;}

.conf-wrap{background:#101520;border:1px solid #1e2330;border-radius:10px;
  padding:1.2rem 1.5rem;margin-top:.8rem;}
.conf-label{font-family:'IBM Plex Mono',monospace;font-size:.62rem;letter-spacing:.1em;
  text-transform:uppercase;color:#6b7591;margin-bottom:.6rem;}
.conf-row{display:flex;align-items:center;gap:.9rem;margin-bottom:.5rem;}
.conf-name{font-family:'IBM Plex Mono',monospace;font-size:.75rem;width:56px;color:#b0b8cc;}
.conf-track{flex:1;background:#1e2330;border-radius:4px;height:8px;overflow:hidden;}
.conf-fill-h{height:100%;border-radius:4px;background:#38c9b0;}
.conf-fill-a{height:100%;border-radius:4px;background:#f06292;}
.conf-pct{font-family:'IBM Plex Mono',monospace;font-size:.75rem;color:#d4d8e2;
  width:46px;text-align:right;}

.ftable{width:100%;border-collapse:collapse;font-size:.82rem;}
.ftable th{font-family:'IBM Plex Mono',monospace;font-size:.62rem;letter-spacing:.1em;
  text-transform:uppercase;color:#6b7591;padding:.5rem .8rem;
  border-bottom:1px solid #1e2330;text-align:left;}
.ftable td{padding:.4rem .8rem;border-bottom:1px solid #1a1f2c;color:#c4c9d4;}
.ftable tr:last-child td{border-bottom:none;}
.ftable .hi{color:#f5a623;font-weight:600;}

.stButton>button{background:#101520!important;border:1px solid #1e2330!important;
  color:#b0b8cc!important;border-radius:7px!important;
  font-family:'IBM Plex Sans',sans-serif!important;font-size:.82rem!important;transition:all .15s!important;}
.stButton>button:hover{border-color:#f5a623!important;color:#f5a623!important;
  background:rgba(245,166,35,.06)!important;}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#c27d10,#a0620a)!important;
  border-color:transparent!important;color:#fff!important;font-weight:600!important;}

.stTabs [data-baseweb="tab-list"]{background:#0f1420;border-bottom:1px solid #1e2330;gap:0;}
.stTabs [data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace;font-size:.72rem;
  letter-spacing:.06em;color:#6b7591;padding:.55rem 1.2rem;border-radius:0;}
.stTabs [aria-selected="true"]{color:#f5a623!important;border-bottom:2px solid #f5a623;}

[data-testid="stExpander"]{background:#101520;border:1px solid #1e2330;border-radius:9px;}
textarea{background:#0f1420!important;border:1px solid #1e2330!important;
  color:#d4d8e2!important;border-radius:8px!important;}
textarea:focus{border-color:#f5a623!important;
  box-shadow:0 0 0 2px rgba(245,166,35,.15)!important;}
hr{border-color:#1e2330!important;}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"#101520","axes.facecolor":"#101520",
    "axes.edgecolor":"#1e2330","axes.labelcolor":"#8892a4",
    "xtick.color":"#6b7591","ytick.color":"#6b7591",
    "text.color":"#d4d8e2","grid.color":"#1e2330",
    "grid.linestyle":"--","grid.alpha":.6,
    "legend.facecolor":"#101520","legend.edgecolor":"#1e2330",
    "font.family":"monospace",
})
PALETTE = {"Human":"#38c9b0","AI":"#f06292"}

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
FEAT_NAMES = [
    "Burstiness (sentence variety)","Avg sentence length",
    "Punctuation diversity","Comma rate",
    "1st-person pronoun %","Academic hedge word %",
    "Unique word ratio (TTR)","Exclamation rate",
    "Question rate","Avg word length",
    "Total text length","Sentence count",
    "Contraction rate","Caps ratio",
]
TOPIC_EMOJIS = {
    "food":"🍕","travel":"✈️","education":"📚","sports":"⚽",
    "finance":"💰","science":"🔬","entertainment":"🎬",
    "lifestyle":"🌿","technology":"💻","health":"🏥",
}

# Clean up opening phrases so the model learns actual writing style
STRIP_PATTERNS = [
    r"^Analysis indicates that \w[\w\s]* is associated with\s*",
    r"^As someone who follows \w+,?\s*",
    r"^I recently experienced \w[\w\s]* in my day-to-day life and found that\s*",
    r"^In my experience,?\s*\w[\w\s]* often leads to\s*",
    r"^After trying several approaches related to \w+,?\s*",
    r"^My personal opinion on \w[\w\s]* is that\s*",
    r"^This article discusses \w+ and highlights that\s*",
    r"^The following summary on \w[\w\s]* shows\s*",
    r"^Research-style summary on \w[\w\s]*:?\s*",
    r"^A concise overview of \w[\w\s]*:?\s*",
]

def strip_template(text: str) -> str:
    t = str(text)
    for p in STRIP_PATTERNS:
        t = re.sub(p, "", t, flags=re.IGNORECASE)
    return t.strip()

# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════
def extract_features(text: str) -> list:
    text = str(text)
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = text.lower().split()
    slen  = [len(s.split()) for s in sents]
    return [
        np.std(slen)/(np.mean(slen)+1e-5) if slen else 0,
        np.mean(slen) if slen else 0,
        len(set(re.findall(r'[^\w\s]',text)))/(len(text)+1e-5)*1000,
        text.count(',')/(len(sents)+1e-5),
        sum(1 for w in words if w in ['i','me','my','mine','myself','we','our','us'])/(len(words)+1e-5)*100,
        sum(1 for w in words if w in ['however','although','therefore','furthermore','moreover','thus','hence'])/(len(words)+1e-5)*100,
        len(set(words))/(len(words)+1e-5),
        text.count('!')/(len(sents)+1e-5),
        text.count('?')/(len(sents)+1e-5),
        np.mean([len(w) for w in words]) if words else 0,
        len(text), len(sents),
        len(re.findall(r"n't|'re|'ve|'ll|'d|'m",text))/(len(words)+1e-5)*100,
        sum(1 for c in text if c.isupper())/(len(text)+1e-5),
    ]

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data(topics_key, min_quality, min_words):
    df = pd.read_csv("ai_vs_human_dataset_medium.csv")
    df.columns = df.columns.str.strip()
    df["Author"] = df["label"].str.strip().str.lower().map(
        {"human":"Human","ai":"AI","0":"Human","1":"AI"})
    df = df.dropna(subset=["text","Author"])
    df["text"] = df["text"].astype(str)
    # ── KEY FIX: strip template openers before any modelling ──
    df["text_clean"] = df["text"].apply(strip_template)
    if list(topics_key):
        df = df[df["topic"].isin(list(topics_key))]
    df = df[df["quality_score"] >= min_quality]
    df = df[df["length_words"]  >= min_words]
    return df.reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")
    k_neighbors  = st.slider("KNN — k neighbors", 1, 15, 5)
    max_features = st.slider("TF-IDF max features", 500, 8000, 4000, step=500)
    test_size    = st.slider("Test split size", 0.10, 0.40, 0.20)

    st.markdown("### 🔍 Data Filters")
    all_topics = ["food","travel","education","sports","finance",
                  "science","entertainment","lifestyle","technology","health"]
    topic_sel = st.multiselect("Filter by topic", all_topics, default=all_topics,
        format_func=lambda t: f"{TOPIC_EMOJIS.get(t,'')} {t.capitalize()}")
    min_quality = st.slider("Min quality score", 1.5, 5.0, 1.5, step=0.1)
    min_words   = st.slider("Min word count", 5, 40, 5)

    st.markdown("### 🎨 Display")
    show_grid = st.toggle("Chart gridlines", value=True)
    plt.rcParams["axes.grid"] = show_grid

active_topics = tuple(sorted(topic_sel if topic_sel else all_topics))
df_raw = load_data(active_topics, min_quality, min_words)

# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">🔬 NLP · Machine Learning · Text Classification</div>
  <h1 class="hero-title">AI <em>vs</em> Human<br>Text Classifier</h1>
  <p class="hero-desc">
    Detects AI-generated vs human-written text using <strong>TF-IDF + 14 Linguistic Features</strong>
    with <strong>K-Nearest Neighbours</strong> and <strong>Logistic Regression</strong>.
    Filter the dataset, train both models, then probe any text live below.
  </p>
</div>
""", unsafe_allow_html=True)


# ── Quick stats ───────────────────────────────────────────────────────────────
vc    = df_raw["Author"].value_counts()
n_h   = vc.get("Human",0); n_a = vc.get("AI",0); n_tot = max(len(df_raw),1)
st.markdown(f"""
<div class="stat-grid">
  <div class="stat amber">
    <div class="stat-label">Total Samples</div>
    <div class="stat-val">{len(df_raw):,}</div>
    <div class="stat-sub">after filters applied</div>
  </div>
  <div class="stat teal">
    <div class="stat-label">Human Texts</div>
    <div class="stat-val">{n_h}</div>
    <div class="stat-sub">{n_h/n_tot*100:.0f}% of dataset</div>
  </div>
  <div class="stat rose">
    <div class="stat-label">AI Texts</div>
    <div class="stat-val">{n_a}</div>
    <div class="stat-sub">{n_a/n_tot*100:.0f}% of dataset</div>
  </div>
  <div class="stat sky">
    <div class="stat-label">Topics</div>
    <div class="stat-val">{df_raw['topic'].nunique()}</div>
    <div class="stat-sub">{', '.join(df_raw['topic'].unique()[:3])}…</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec">01 — Exploratory Data Analysis</div>', unsafe_allow_html=True)

with st.expander("🗂️ Browse & Explore Dataset", expanded=True):
    eda1, eda2, eda3, eda4 = st.tabs([
        "📋 Sample Rows","📊 Distributions","📌 Topic Breakdown","🤖 AI Source Detail"
    ])

    with eda1:
        fc1, fc2, fc3 = st.columns([1,1,2])
        auth_f = fc1.selectbox("Label",  ["All","Human","AI"], key="s_auth")
        top_f  = fc2.selectbox("Topic",  ["All"]+sorted(df_raw["topic"].unique().tolist()), key="s_top")
        n_rows = fc3.slider("Rows to show", 5, 50, 10, key="s_n")
        dfs = df_raw.copy()
        if auth_f != "All": dfs = dfs[dfs["Author"]==auth_f]
        if top_f  != "All": dfs = dfs[dfs["topic"] ==top_f]
        st.dataframe(
            dfs[["id","Author","topic","text","quality_score","sentiment","plagiarism_score","source_detail"]]
              .sample(min(n_rows, max(len(dfs),1)), random_state=1),
            use_container_width=True, hide_index=True)

    with eda2:
        fig_d, axes_d = plt.subplots(1, 4, figsize=(16, 3.5))
        for ax, col, lbl in [
            (axes_d[0],"quality_score","Quality Score"),
            (axes_d[1],"sentiment","Sentiment"),
            (axes_d[2],"plagiarism_score","Plagiarism Score"),
            (axes_d[3],"length_words","Word Count"),
        ]:
            for auth, clr in PALETTE.items():
                sub = df_raw[df_raw["Author"]==auth][col].dropna()
                ax.hist(sub, bins=20, alpha=.65, color=clr, label=auth, edgecolor="none")
            ax.set_title(lbl, fontsize=9, pad=6); ax.set_xlabel(lbl, fontsize=8)
        p = [mpatches.Patch(color=c,label=l) for l,c in PALETTE.items()]
        axes_d[0].legend(handles=p, fontsize=7, framealpha=.3)
        plt.tight_layout(); st.pyplot(fig_d); plt.close(fig_d)

    with eda3:
        tc = df_raw.groupby(["topic","Author"]).size().unstack(fill_value=0)
        ts = tc.sum(axis=1).sort_values(ascending=True).index
        fig_t, ax_t = plt.subplots(figsize=(9,4))
        y = np.arange(len(ts)); bh = .38
        for i,(auth,clr) in enumerate(PALETTE.items()):
            vals = [tc.loc[t,auth] if auth in tc.columns else 0 for t in ts]
            ax_t.barh(y+(i-.5)*bh, vals, height=bh, color=clr, label=auth, alpha=.85, edgecolor="none")
        ax_t.set_yticks(y)
        ax_t.set_yticklabels([f"{TOPIC_EMOJIS.get(t,'')} {t.capitalize()}" for t in ts], fontsize=8)
        ax_t.set_xlabel("Count",fontsize=8); ax_t.set_title("Texts per Topic by Author Type",fontsize=10,pad=8)
        p = [mpatches.Patch(color=c,label=l) for l,c in PALETTE.items()]
        ax_t.legend(handles=p,fontsize=8,framealpha=.3)
        plt.tight_layout(); st.pyplot(fig_t); plt.close(fig_t)

    with eda4:
        ai_df = df_raw[df_raw["Author"]=="AI"]
        src   = ai_df["source_detail"].value_counts()
        fig_s, ax_s = plt.subplots(figsize=(7,3.5))
        cs = plt.cm.plasma(np.linspace(.2,.85,len(src)))
        bars_s = ax_s.barh(src.index[::-1], src.values[::-1], color=cs[::-1], edgecolor="none", alpha=.85)
        ax_s.set_xlabel("Count",fontsize=8); ax_s.set_title("AI Texts by Source Model",fontsize=10,pad=8)
        for bar, v in zip(bars_s, src.values[::-1]):
            ax_s.text(v+.15, bar.get_y()+bar.get_height()/2, str(v), va="center", fontsize=8)
        plt.tight_layout()
        cs2, _ = st.columns([2,1])
        with cs2: st.pyplot(fig_s); plt.close(fig_s)
        q_src = ai_df.groupby("source_detail")["quality_score"].agg(["mean","std","count"]).round(2)
        q_src.columns = ["Mean Quality","Std Dev","Count"]
        st.dataframe(q_src.sort_values("Mean Quality",ascending=False), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec">02 — Model Training & Evaluation</div>', unsafe_allow_html=True)

if len(df_raw) < 20:
    st.error("Not enough samples after filtering. Please loosen the sidebar filters.")
    st.stop()

col_btn, col_info = st.columns([1,3])
with col_btn:
    train_btn = st.button("🚀 Train Models", type="primary", use_container_width=True)
with col_info:
    st.caption(f"Training on **{len(df_raw):,}** samples "
               f"→ {int(len(df_raw)*(1-test_size))} train / {int(len(df_raw)*test_size)} test")

# Initialise every key that results-rendering will read.
# This guards against stale session state from a previous app version.
_defaults = {
    "models_trained": False,
    "knn": None, "lr": None, "tfidf": None, "scaler": None, "le": None,
    "acc_knn": None, "acc_lr": None,
    "cv_lr": None, "cv_knn": None,
    "y_test": None, "y_pred_knn": None, "y_pred_lr": None,
    "X_test": None, "human_label": None, "demo_text": "",
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if train_btn:
    prog = st.progress(0, text="Encoding labels…")
    le   = LabelEncoder()
    y    = le.fit_transform(df_raw["Author"])
    prog.progress(15, text="Vectorising cleaned text with TF-IDF…")

    tfidf   = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(df_raw["text_clean"])   # ← uses stripped text
    prog.progress(35, text="Extracting 14 linguistic features…")

    X_feat        = np.array([extract_features(t) for t in df_raw["text_clean"]])
    scaler        = StandardScaler()
    X_feat_scaled = scaler.fit_transform(X_feat)
    X_all         = hstack([X_tfidf, csr_matrix(X_feat_scaled)])
    prog.progress(55, text="Splitting train / test…")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=test_size, random_state=42, stratify=y)

    prog.progress(65, text="Training KNN…")
    knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric="euclidean")
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc_knn    = accuracy_score(y_test, y_pred_knn)

    prog.progress(80, text="Training Logistic Regression…")
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    lr.fit(X_train, y_train)
    y_pred_lr  = lr.predict(X_test)
    acc_lr     = accuracy_score(y_test, y_pred_lr)

    prog.progress(92, text="Running 5-fold cross-validation…")
    cv_lr  = cross_val_score(lr,  X_all, y, cv=5, scoring="accuracy")
    cv_knn = cross_val_score(knn, X_all, y, cv=5, scoring="accuracy")

    human_label = next((c for c in le.classes_ if c.lower()=="human"), le.classes_[0])
    prog.progress(100, text="Done!"); prog.empty()

    st.session_state.update({
        "models_trained": True,
        "knn":knn,"lr":lr,"tfidf":tfidf,"scaler":scaler,"le":le,
        "acc_knn":acc_knn,"acc_lr":acc_lr,
        "cv_lr":cv_lr,"cv_knn":cv_knn,
        "y_test":y_test,"y_pred_knn":y_pred_knn,"y_pred_lr":y_pred_lr,
        "X_test":X_test,"human_label":human_label,
    })
    st.success("✅ Both models trained successfully!")

if st.session_state.models_trained:
    acc_knn    = st.session_state.acc_knn
    acc_lr     = st.session_state.acc_lr
    cv_lr      = st.session_state.cv_lr
    cv_knn     = st.session_state.cv_knn
    y_test     = st.session_state.y_test
    y_pred_knn = st.session_state.y_pred_knn
    y_pred_lr  = st.session_state.y_pred_lr
    le         = st.session_state.le
    X_test     = st.session_state.X_test
    best       = "LR" if acc_lr >= acc_knn else "KNN"
    delta_pp   = abs(acc_lr-acc_knn)*100

    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat amber">
        <div class="stat-label">KNN Accuracy</div>
        <div class="stat-val">{acc_knn:.4f}</div>
        <div class="stat-sub">+{(acc_knn-.5)*100:.1f}pp above chance</div>
      </div>
      <div class="stat teal">
        <div class="stat-label">LR Accuracy</div>
        <div class="stat-val">{acc_lr:.4f}</div>
        <div class="stat-sub">+{(acc_lr-.5)*100:.1f}pp above chance</div>
      </div>
      <div class="stat rose">
        <div class="stat-label">Best Model</div>
        <div class="stat-val">{best}</div>
        <div class="stat-sub">leads by {delta_pp:.2f}pp</div>
      </div>
      <div class="stat sky">
        <div class="stat-label">Test Samples</div>
        <div class="stat-val">{len(y_test)}</div>
        <div class="stat-sub">{test_size*100:.0f}% held-out set</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    ev1, ev2, ev3, ev4, ev5, ev6 = st.tabs([
        "📋 Classification Report",
        "📊 Accuracy Chart",
        "📉 Cross-Validation",
        "🔲 Confusion Matrix",
        "📈 ROC Curve",
        "🔵 PCA Viz",
    ])

    with ev1:
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**🔵 KNN**")
            st.code(classification_report(y_test, y_pred_knn, target_names=le.classes_), language=None)
        with r2:
            st.markdown("**🟢 Logistic Regression**")
            st.code(classification_report(y_test, y_pred_lr, target_names=le.classes_), language=None)

    with ev2:
        fig_b, ax_b = plt.subplots(figsize=(5,3.2))
        bars_b = ax_b.bar(["KNN","Logistic\nRegression"], [acc_knn,acc_lr],
                          color=["#f5a623","#38c9b0"], width=.4, edgecolor="none")
        lo = max(0, min(acc_knn,acc_lr)-.15)
        ax_b.set_ylim(lo, min(1.04, max(acc_knn,acc_lr)+.08))
        ax_b.set_ylabel("Accuracy",fontsize=9)
        ax_b.set_title("Model Accuracy Comparison",fontsize=10,pad=8)
        for bar,v in zip(bars_b,[acc_knn,acc_lr]):
            ax_b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.003,
                      f"{v:.4f}", ha="center",va="bottom",fontsize=10,fontweight="bold")
        plt.tight_layout()
        cb,_ = st.columns([1,1])
        with cb: st.pyplot(fig_b); plt.close(fig_b)

    with ev3:
        if cv_knn is not None and cv_lr is not None:
            fig_cv, ax_cv = plt.subplots(figsize=(7,3.5))
            x = np.arange(5); w = .35
            ax_cv.bar(x-w/2, cv_knn, width=w, color="#f5a623", alpha=.85, label=f"KNN (mean={cv_knn.mean():.3f})", edgecolor="none")
            ax_cv.bar(x+w/2, cv_lr,  width=w, color="#38c9b0", alpha=.85, label=f"LR  (mean={cv_lr.mean():.3f})", edgecolor="none")
            ax_cv.set_xticks(x); ax_cv.set_xticklabels([f"Fold {i+1}" for i in range(5)], fontsize=8)
            ax_cv.set_ylabel("Accuracy",fontsize=9)
            ax_cv.set_title("5-Fold Cross-Validation Scores",fontsize=10,pad=8)
            ax_cv.axhline(cv_knn.mean(), color="#f5a623", lw=1, linestyle="--", alpha=.5)
            ax_cv.axhline(cv_lr.mean(),  color="#38c9b0", lw=1, linestyle="--", alpha=.5)
            ax_cv.set_ylim(max(0, min(cv_knn.min(),cv_lr.min())-.1), min(1.05, max(cv_knn.max(),cv_lr.max())+.07))
            ax_cv.legend(fontsize=8,framealpha=.3)
            plt.tight_layout()
            st.pyplot(fig_cv); plt.close(fig_cv)
            st.caption("Cross-validation uses all data, giving a more reliable estimate of generalisation performance.")
        else:
            st.info("Train the models to see cross-validation scores.")

    with ev4:
        fig_cm,(ax_k,ax_l) = plt.subplots(1,2,figsize=(10,4))
        for ax, ypred, title, cmap in [
            (ax_k, y_pred_knn, f"KNN (k={k_neighbors})", "YlOrBr"),
            (ax_l, y_pred_lr,  "Logistic Regression",    "BuGn"),
        ]:
            cm = confusion_matrix(y_test, ypred)
            im = ax.imshow(cm, cmap=cmap, aspect="auto")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(le.classes_); ax.set_yticklabels(le.classes_)
            ax.set_xlabel("Predicted",fontsize=8); ax.set_ylabel("Actual",fontsize=8)
            ax.set_title(title,fontsize=10)
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=14,fontweight="bold",
                            color="white" if cm[i,j]>cm.max()*.55 else "#a0aec0")
            plt.colorbar(im,ax=ax,shrink=.85)
        plt.tight_layout(); st.pyplot(fig_cm); plt.close(fig_cm)

    with ev5:
        h_idx   = list(le.classes_).index(st.session_state.human_label)
        lr_prob = st.session_state.lr.predict_proba(X_test)[:,h_idx]
        fpr,tpr,_ = roc_curve(y_test, lr_prob, pos_label=h_idx)
        roc_auc   = auc(fpr,tpr)
        fig_r, ax_r = plt.subplots(figsize=(5,4))
        ax_r.plot(fpr,tpr,color="#38c9b0",lw=2,label=f"LR ROC (AUC={roc_auc:.3f})")
        ax_r.plot([0,1],[0,1],color="#6b7591",lw=1,linestyle="--",label="Random (0.5)")
        ax_r.fill_between(fpr,tpr,alpha=.08,color="#38c9b0")
        ax_r.set_xlabel("False Positive Rate",fontsize=9)
        ax_r.set_ylabel("True Positive Rate",fontsize=9)
        ax_r.set_title("ROC Curve — Logistic Regression",fontsize=10,pad=8)
        ax_r.legend(fontsize=8,framealpha=.3)
        plt.tight_layout()
        cr,_ = st.columns([1,1])
        with cr: st.pyplot(fig_r); plt.close(fig_r)
        st.caption(f"AUC = {roc_auc:.4f} · 1.0 = perfect · 0.5 = random")

    with ev6:
        pca   = PCA(n_components=2)
        n_s   = min(500, X_test.shape[0])
        rng   = np.random.default_rng(seed=7)
        idx   = rng.choice(X_test.shape[0], n_s, replace=False)
        Xd    = X_test[idx].toarray() if hasattr(X_test[idx],"toarray") else X_test[idx]
        Xp    = pca.fit_transform(Xd)
        pca_df = pd.DataFrame(Xp,columns=["PC1","PC2"])
        pca_df["True"] = le.inverse_transform(y_test[idx])
        pca_df["KNN"]  = le.inverse_transform(y_pred_knn[idx])
        pca_df["LR"]   = le.inverse_transform(y_pred_lr[idx])
        fig_p, axes_p = plt.subplots(1,3,figsize=(16,4.5))
        for ax,col,title in zip(axes_p,["True","KNN","LR"],
                                ["True Labels","KNN Predictions","LR Predictions"]):
            for auth,clr in PALETTE.items():
                m = pca_df[col]==auth
                ax.scatter(pca_df.loc[m,"PC1"],pca_df.loc[m,"PC2"],
                           c=clr,label=auth,alpha=.6,s=18,edgecolors="none")
            ax.set_title(title,fontsize=9,pad=6)
            ax.set_xlabel("PC1",fontsize=8); ax.set_ylabel("PC2",fontsize=8)
            p = [mpatches.Patch(color=c,label=l) for l,c in PALETTE.items()]
            ax.legend(handles=p,fontsize=7,framealpha=.3)
        plt.tight_layout(); st.pyplot(fig_p); plt.close(fig_p)
        vr = pca.explained_variance_ratio_
        st.caption(f"PC1 = {vr[0]*100:.1f}% variance · PC2 = {vr[1]*100:.1f}%")

    with st.expander("📌 LR Feature Importance — Top Discriminative Features", expanded=False):
        coef      = st.session_state.lr.coef_[0]
        vocab     = st.session_state.tfidf.get_feature_names_out().tolist()
        all_names = vocab + FEAT_NAMES
        n_c       = min(len(all_names), len(coef))
        coef_df   = pd.DataFrame({"feature":all_names[:n_c],"coef":coef[:n_c]})
        combined  = pd.concat([coef_df.nlargest(15,"coef"), coef_df.nsmallest(15,"coef")]).sort_values("coef")
        fig_i, ax_i = plt.subplots(figsize=(8,5))
        ci = ["#38c9b0" if v<0 else "#f06292" for v in combined["coef"]]
        ax_i.barh(combined["feature"], combined["coef"], color=ci, edgecolor="none", alpha=.85)
        ax_i.axvline(0,color="#6b7591",lw=.8)
        ax_i.set_xlabel("LR Coefficient  (positive → AI,  negative → Human)",fontsize=8)
        ax_i.set_title("Top 30 Discriminative Features",fontsize=9,pad=8)
        ax_i.tick_params(axis="y",labelsize=7)
        p = [mpatches.Patch(color="#38c9b0",label="→ Human"),
             mpatches.Patch(color="#f06292",label="→ AI")]
        ax_i.legend(handles=p,fontsize=7,framealpha=.3)
        plt.tight_layout(); st.pyplot(fig_i); plt.close(fig_i)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.models_trained:
    st.markdown('<div class="sec">03 — Live Text Prediction</div>', unsafe_allow_html=True)
    st.markdown("Enter any text below — casual, formal, academic, or social-media style. Both models classify it instantly.")

    DEMO_TEXTS = {
        "🧑 Casual":    "I tried cooking a new recipe today and it turned out surprisingly good. My friends loved it!",
        "🧑 Personal":  "Traveling always teaches you something. I love exploring new places on my own terms.",
        "🧑 Emotional": "Grief is strange. It comes in waves and catches you off guard sometimes. I miss him every day.",
        "🤖 Tech AI":   "Machine learning algorithms analyze patterns in data to make predictions and informed decisions.",
        "🤖 Academic":  "The proposed methodology demonstrates superior performance compared to existing baseline approaches.",
        "🤖 Formal AI": "Costs can vary greatly depending on implementation. Community feedback will shape future developments.",
    }
    demo_cols = st.columns(3)
    for i,(label,text) in enumerate(DEMO_TEXTS.items()):
        with demo_cols[i%3]:
            if st.button(label, use_container_width=True, key=f"demo_{i}"):
                st.session_state.demo_text = text

    user_input = st.text_area("Enter text to classify:",
        value=st.session_state.get("demo_text",""),
        height=120, placeholder="Paste or type any text here…")

    wc = len(user_input.split()) if user_input else 0
    st.caption(f"Word count: **{wc}** {'✅' if wc>=5 else '⚠️  — need at least 5 words'}")

    if user_input and wc >= 5:
        cleaned     = strip_template(user_input)
        feat        = np.array([extract_features(cleaned)])
        feat_scaled = st.session_state.scaler.transform(feat)
        tfidf_vec   = st.session_state.tfidf.transform([cleaned])
        combined    = hstack([tfidf_vec, csr_matrix(feat_scaled)])

        knn_res     = st.session_state.le.inverse_transform(st.session_state.knn.predict(combined))[0]
        lr_res      = st.session_state.le.inverse_transform(st.session_state.lr.predict(combined))[0]
        human_label = st.session_state.human_label

        lr_proba    = st.session_state.lr.predict_proba(combined)[0]
        classes     = list(st.session_state.le.classes_)
        h_idx2      = classes.index(human_label) if human_label in classes else 0
        human_conf  = lr_proba[h_idx2]*100
        ai_conf     = 100-human_conf


        rc1, rc2 = st.columns(2)
        for col, model_name, res in [(rc1,"KNN",knn_res),(rc2,"Logistic Regression",lr_res)]:
            with col:
                is_h = res==human_label
                st.markdown(f"""
                <div class="res-card {'res-human' if is_h else 'res-ai'}">
                  <div class="res-model">{model_name}</div>
                  <div class="res-verdict {'col-human' if is_h else 'col-ai'}">
                    {'🧑 Human Written' if is_h else '🤖 AI Generated'}
                  </div>
                </div>""", unsafe_allow_html=True)

        agree = "✅ Both models agree" if knn_res==lr_res else "⚡ Models disagree"
        st.markdown(f"""
        <div class="conf-wrap">
          <div class="conf-label">LR Confidence — {agree}</div>
          <div class="conf-row">
            <div class="conf-name">Human</div>
            <div class="conf-track"><div class="conf-fill-h" style="width:{human_conf:.1f}%"></div></div>
            <div class="conf-pct">{human_conf:.1f}%</div>
          </div>
          <div class="conf-row">
            <div class="conf-name">AI</div>
            <div class="conf-track"><div class="conf-fill-a" style="width:{ai_conf:.1f}%"></div></div>
            <div class="conf-pct">{ai_conf:.1f}%</div>
          </div>
        </div>""", unsafe_allow_html=True)

        fig_c, ax_c = plt.subplots(figsize=(6,2))
        ax_c.barh(["AI","Human"],[ai_conf,human_conf],
                  color=["#f06292","#38c9b0"],height=.42,edgecolor="none")
        ax_c.set_xlim(0,100); ax_c.axvline(50,color="#6b7591",linestyle="--",lw=.9)
        for i,v in enumerate([ai_conf,human_conf]):
            ax_c.text(min(v+1.5,90),i,f"{v:.1f}%",va="center",fontsize=9,fontweight="bold")
        ax_c.set_xlabel("Confidence (%)",fontsize=8)
        ax_c.set_title("Prediction Confidence",fontsize=9,pad=6)
        plt.tight_layout()
        cc,_ = st.columns([1,1])
        with cc: st.pyplot(fig_c); plt.close(fig_c)

        with st.expander("🔬 Linguistic Feature Breakdown", expanded=False):
            feat_df = pd.DataFrame({"Feature":FEAT_NAMES,"Value":np.round(feat[0],4)})
            feat_df = feat_df.sort_values("Value",key=abs,ascending=False)
            med = feat_df["Value"].abs().median()
            rows_html = "".join(
                f'<tr><td>{r["Feature"]}</td><td class="{"hi" if abs(r["Value"])>med else ""}">{r["Value"]}</td></tr>'
                for _,r in feat_df.iterrows()
            )
            st.markdown(f"""
            <table class="ftable">
              <thead><tr><th>Feature</th><th>Value</th></tr></thead>
              <tbody>{rows_html}</tbody>
            </table>""", unsafe_allow_html=True)
            st.markdown("""
            > ↑ 1st-person pronouns · contractions · exclamations · burstiness = 🧑 **Human**
            > ↑ Hedge words · avg word length · low burstiness · long sentences = 🤖 **AI**
            """)
    elif user_input and wc < 5:
        st.warning("⚠️ Please enter at least 5 words for a meaningful prediction.")

# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center;font-family:"IBM Plex Mono",monospace;font-size:.65rem;
            color:#2a3050;padding:.8rem 0 1.2rem;letter-spacing:.08em;'>
  AI vs Human Text Classifier · TF-IDF + Linguistic Features · KNN + Logistic Regression
</div>
""", unsafe_allow_html=True)