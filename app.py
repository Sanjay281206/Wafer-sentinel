"""WAFER SENTINEL - premium Streamlit semiconductor intelligence application."""
import io
import json
import os
import platform
import subprocess
import sys
import html
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import psutil
except Exception:
    psutil = None

from wafersentinel.classes import PATTERNS
from wafersentinel.data import resize_map
from wafersentinel.gradcam import gradcam
from wafersentinel.pipeline import Sentinel
from wafersentinel.synth import make_map
from wafersentinel.yield_calc import radial_profile

st.set_page_config(
    page_title="WAFER SENTINEL",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');
:root {
  --navy:#07111f; --panel:#101d30; --panel2:#14263d; --cyan:#32d9f5;
  --blue:#477bff; --violet:#9b7cff; --green:#31d69a; --amber:#ffb84d;
  --red:#ff5577; --text:#f1f6ff; --muted:#9fb0c7;
}
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
.stApp {
  background:
    linear-gradient(rgba(50,217,245,.018) 1px,transparent 1px),
    linear-gradient(90deg,rgba(50,217,245,.018) 1px,transparent 1px),
    radial-gradient(circle at 72% 0%,#102642 0,#091526 36%,#07111f 72%);
  background-size:42px 42px,42px 42px,100% 100%;
  color:var(--text);
}
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1520px; padding:1.1rem 2.2rem 3rem; }
[data-testid="stSidebar"] { background:#091525; border-right:1px solid #20344d; }
.brand-header { position:relative; text-align:center; padding:.25rem 0 1.05rem; overflow:hidden; }
.brand-header:after {
  content:""; position:absolute; left:-10%; right:-10%; bottom:.55rem; height:1px;
  background:linear-gradient(90deg,transparent,rgba(50,217,245,.5),transparent);
  animation:scan 7s linear infinite;
}
@keyframes scan { 0%{transform:translateX(-55%)} 100%{transform:translateX(55%)} }
.logo-row { display:flex; align-items:center; justify-content:center; gap:.7rem; }
.wafer-logo {
  width:46px;height:46px;border:1px solid rgba(50,217,245,.7);border-radius:50%;
  display:grid;place-items:center;color:var(--cyan);font-weight:800;
  box-shadow:inset 0 0 18px rgba(50,217,245,.09),0 0 22px rgba(50,217,245,.07);
  background:radial-gradient(circle,#14263d 0 44%,#0a1a2d 45% 100%);
}
.brand-title {
  margin:0;font-family:'Space Grotesk','Inter',sans-serif;font-size:2.28rem;line-height:1;
  letter-spacing:.15em;font-weight:800;
  background:linear-gradient(90deg,#dffaff 5%,#32d9f5 48%,#477bff 90%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.brand-subtitle { margin-top:.42rem;color:#a9bad1;font-size:.82rem;font-weight:600;letter-spacing:.025em; }
.glass-panel {
  border:1px solid rgba(99,151,190,.24);border-radius:15px;
  background:linear-gradient(145deg,rgba(20,38,61,.94),rgba(12,27,45,.92));
  box-shadow:0 16px 38px rgba(0,0,0,.19),inset 0 1px rgba(255,255,255,.025);
  padding:1.15rem;transition:.18s ease;
}
.glass-panel:hover { transform:translateY(-2px);border-color:rgba(50,217,245,.36); }
.panel-title {
  color:#dfeaff;font-size:.78rem;text-transform:uppercase;letter-spacing:.13em;
  font-weight:800;margin-bottom:.75rem;
}
.start-title { font-size:1.3rem;font-weight:800;text-align:center;color:#f4f8ff;margin:.15rem 0 .25rem; }
.start-desc { text-align:center;color:#9fb0c7;font-size:.86rem;margin-bottom:.9rem; }
.brand-subtitle, .panel-title, .start-title, .start-desc,
h1,h2,h3,h4,h5,h6 { font-family:'Space Grotesk','Inter',sans-serif; }
.kpi-value, [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
.status-chip, code, pre, [data-testid="stDataFrame"], .insight-card b,
.small-note { font-family:'IBM Plex Mono','Inter',sans-serif; }
.kpi-card {
  min-height:105px;border:1px solid #264158;border-radius:13px;padding:.9rem 1rem;
  background:linear-gradient(145deg,rgba(20,38,61,.96),rgba(11,25,42,.96));
  box-shadow:0 12px 28px rgba(0,0,0,.16);
}
.kpi-label { color:#91a5bf;font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;font-weight:700; }
.kpi-value { color:#f5f9ff;font-size:1.5rem;font-weight:800;margin:.38rem 0 .12rem; }
.kpi-sub { color:#91a5bf;font-size:.72rem; }
.insight-card {
  border-left:3px solid var(--cyan);background:#0d1b2c;border-radius:9px;
  padding:.8rem .9rem;margin:.45rem 0;color:#dae6f5;
}
.insight-card.amber { border-left-color:var(--amber); }
.insight-card.green { border-left-color:var(--green); }
.insight-card.red { border-left-color:var(--red); }
.insight-card b { color:#edf5ff;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em; }
.status-chip {
  display:inline-block;border:1px solid #2b4561;border-radius:999px;padding:.28rem .58rem;
  color:#bfccdc;background:#0d1a2b;font-size:.68rem;font-weight:700;margin:.12rem;
}
.status-chip.good { color:#79deb3;border-color:rgba(49,214,154,.42); }
.status-chip.warn { color:#ffc875;border-color:rgba(255,184,77,.42); }
.status-chip.bad { color:#ff91a6;border-color:rgba(255,85,119,.42); }
.section-head { display:flex;align-items:center;justify-content:space-between;margin:.2rem 0 .8rem; }
.section-head h3 { margin:0;font-size:1.25rem; }
.root-hero {
  border:1px solid rgba(50,217,245,.30);border-radius:14px;padding:1rem 1.15rem;
  background:linear-gradient(145deg,rgba(20,38,61,.98),rgba(10,24,40,.96));
  box-shadow:0 18px 38px rgba(0,0,0,.24),inset 0 1px rgba(255,255,255,.04);
}
.root-grid-card {
  min-height:170px;border:1px solid #28465f;border-radius:12px;padding:.95rem 1rem;
  background:linear-gradient(145deg,#14263d,#0d1b2c);
  box-shadow:0 14px 28px rgba(0,0,0,.18),inset 0 1px rgba(255,255,255,.025);
}
.root-grid-card h4 {
  margin:0 0 .65rem;font-family:'Space Grotesk','Inter',sans-serif;
  color:#32d9f5;font-size:.76rem;text-transform:uppercase;letter-spacing:.11em;
}
.root-grid-card p,.root-grid-card li { color:#e3edf8;font-family:'Inter',sans-serif;font-size:.82rem;line-height:1.58; }
.root-grid-card ul { margin:.15rem 0 0;padding-left:1.05rem; }
.muted { color:var(--muted); }
.stTabs [data-baseweb="tab-list"] { gap:.35rem;background:transparent;border-bottom:1px solid #263d55;padding-bottom:.35rem; }
.stTabs [data-baseweb="tab"] { height:44px;border-radius:8px 8px 0 0;padding:.5rem .8rem;color:#aec0d5; }
.stTabs [aria-selected="true"] { background:#14263d;color:#edf7ff;border-bottom:2px solid #32d9f5; }
[data-testid="stMetric"] {
  background:linear-gradient(145deg,#14263d,#0d1b2c);border:1px solid #264158;border-radius:11px;padding:.75rem;
}
[data-testid="stMetricLabel"],[data-testid="stMetricValue"] { color:#eaf3ff !important; }
[data-testid="stFileUploader"] section { background:#0d1b2c;border:1px dashed #477bff;border-radius:12px; }
[data-testid="stFileUploader"] * { color:#d7e3f2 !important; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background:#0d1b2c !important;border-color:#294764 !important;color:#edf5ff !important; }
div[data-baseweb="select"] span, input { color:#edf5ff !important; }

/* Selected checkbox and multiselect wording: white controls with black text */
div[data-baseweb="popover"] *,
[role="combobox"] *,
[role="searchbox"] *,
[data-baseweb="menu"] *,
[data-baseweb="menu"] [role="option"] *,
[data-baseweb="menu"] [role="option"] span,
[data-baseweb="menu"] [role="option"] div {
  color:#111827 !important;
}
div[data-baseweb="popover"],
[data-baseweb="menu"] {
  background:#ffffff !important;
  color:#111827 !important;
}
[role="combobox"],
div[data-baseweb="select"] input,
div[data-baseweb="input"] input {
  background:#ffffff !important;
  color:#111827 !important;
  -webkit-text-fill-color:#111827 !important;
  caret-color:#111827 !important;
}
[role="combobox"]::placeholder,
div[data-baseweb="select"] input::placeholder,
div[data-baseweb="input"] input::placeholder {
  color:#4b5563 !important;
  -webkit-text-fill-color:#4b5563 !important;
  opacity:1;
}
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] p,
[data-testid="stCheckbox"] span { color:#111827 !important; }
[data-testid="stCheckbox"] label {
  background:#ffffff;border:1px solid #d7e1ea;border-radius:8px;
  padding:.42rem .58rem;
}
[data-testid="stCheckbox"] div[role="checkbox"] {
  background:#ffffff !important;border:2px solid #111827 !important;
  box-shadow:none !important;
}
[data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {
  background:#ffffff !important;border-color:#111827 !important;
}
[data-baseweb="checkbox"] {
  background:#ffffff !important;border:1px solid #111827 !important;
}
[data-baseweb="checkbox"] * { color:#111827 !important; }
div[data-baseweb="tag"] {
  background:#ffffff !important;border:1px solid #d7e1ea !important;
  border-radius:7px !important;
}
div[data-baseweb="tag"] span,
div[data-baseweb="tag"] div,
div[data-baseweb="tag"] svg { color:#111827 !important;fill:#111827 !important; }

/* Readable radio, dropdown and select-option wording */
[data-testid="stRadio"] label, [data-testid="stRadio"] p,
[data-testid="stRadio"] span { color:#edf5ff !important; }
div[role="listbox"] { background:#ffffff !important; }
div[role="option"] { background:#ffffff !important;color:#111827 !important; }
div[role="option"] * { color:#111827 !important; }
.stButton button {
  width:100%;border:1px solid #32d9f5;border-radius:11px;
  background:linear-gradient(90deg,#1479a5,#477bff);color:#ffffff !important;
  font-weight:800;letter-spacing:.09em;padding:.72rem 1rem;
  box-shadow:0 10px 24px rgba(71,123,255,.16);
}
.stButton button:hover { border-color:#85eaff;box-shadow:0 12px 28px rgba(50,217,245,.20); }
.stDownloadButton button { background:#14263d;border:1px solid #477bff;color:#edf5ff; }
.stExpander { border:1px solid #264158;border-radius:10px;background:#0d1b2c; }
[data-testid="stNotification"],.stAlert { background:#10213a !important;border:1px solid #2d4a66 !important;color:#edf5ff !important; }
h1,h2,h3,h4,h5,h6,p,label,span,small { color:#edf5ff; }
.small-note { color:#96a8c0;font-size:.76rem;line-height:1.55; }
hr { border-color:#23384f; }
@media (max-width:900px) { .block-container{padding:1rem} .brand-title{font-size:1.55rem} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def init_state():
    defaults = {
        "analysis": None,
        "map": None,
        "processed_map": None,
        "wafer_id": None,
        "dataset_source": None,
        "input_source": None,
        "history": [],
        "gradcam": None,
        "uploaded_name": None,
        "uploaded_bytes": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def plot_layout(fig, height=420):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=30, r=20, t=45, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,17,31,.78)",
        font=dict(color="#dce7f5", family="Inter, Arial"),
        hoverlabel=dict(bgcolor="#14263d", bordercolor="#32d9f5", font_color="#f5f9ff"),
        legend=dict(orientation="h", y=1.02),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(105,145,180,.12)",
        zeroline=False,
        title_font=dict(family="IBM Plex Mono, Inter, Arial", color="#dce7f5"),
        tickfont=dict(family="IBM Plex Mono, Inter, Arial", color="#b7c7d8"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(105,145,180,.12)",
        zeroline=False,
        title_font=dict(family="IBM Plex Mono, Inter, Arial", color="#dce7f5"),
        tickfont=dict(family="IBM Plex Mono, Inter, Arial", color="#b7c7d8"),
    )
    return fig


def wafer_figure(m, title="Wafer Map", center_line=True):
    n0, n1 = m.shape
    y, x = np.mgrid[0:n0, 0:n1]
    fig = go.Figure()
    for mask, name, color, size in (
        (m == 0, "Unavailable", "#26384d", 5),
        (m == 1, "Good die", "#31d69a", 6),
        (m == 2, "Defective die", "#ff5577", 7),
    ):
        fig.add_trace(go.Scatter(
            x=x[mask], y=y[mask], mode="markers", name=name,
            marker=dict(size=size, color=color, line=dict(width=0)),
            customdata=np.full(int(mask.sum()), name),
            hovertemplate="X %{x}<br>Y %{y}<br>Status %{customdata}<extra></extra>",
        ))
    if center_line:
        cy, cx = (n0 - 1) / 2, (n1 - 1) / 2
        rr = min(n0, n1) / 2
        theta = np.linspace(0, 2*np.pi, 300)
        fig.add_trace(go.Scatter(
            x=cx + rr*np.cos(theta), y=cy + rr*np.sin(theta),
            mode="lines", line=dict(color="#32d9f5", width=1.5),
            showlegend=False, hoverinfo="skip",
        ))
        fig.update_xaxes(range=[-2, n1+1])
        fig.update_yaxes(range=[n0+1, -2])
    fig.update_layout(title=title)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return plot_layout(fig, 470)


def radial_figure(m):
    profile = radial_profile(m)
    fig = go.Figure(go.Bar(
        x=[f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)],
        y=np.array(profile)*100,
        marker=dict(color=np.array(profile), colorscale=[[0,"#31d69a"],[.5,"#ffb84d"],[1,"#ff5577"]]),
        hovertemplate="Radius band %{x}<br>Failure %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(title="Radial Defect Profile", yaxis_title="Failure rate (%)")
    return plot_layout(fig, 330)


def zone_figure(stats):
    z = stats["zone_fail_rate"]
    fig = go.Figure(go.Bar(
        x=["Center","Middle","Edge"],
        y=[z["center"]*100,z["mid"]*100,z["edge"]*100],
        marker=dict(color=["#32d9f5","#477bff","#ff5577"]),
        text=[f"{v:.1f}%" for v in (z["center"]*100,z["mid"]*100,z["edge"]*100)],
        textposition="outside",
    ))
    fig.update_layout(title="Region Failure Rate", yaxis_title="Failure rate (%)")
    fig.update_yaxes(rangemode="tozero")
    return plot_layout(fig, 330)


def bar_figure(values, title, y_title="Probability"):
    fig = go.Figure(go.Bar(
        x=list(values.keys()), y=list(values.values()),
        marker=dict(color=["#32d9f5","#477bff","#9b7cff","#31d69a","#ffb84d","#ff5577","#5ed6b2","#b09bff"][:len(values)]),
    ))
    fig.update_layout(title=title, yaxis_title=y_title)
    return plot_layout(fig, 330)


def defect_3d_figure(m):
    y, x = np.mgrid[0:m.shape[0],0:m.shape[1]]
    fail = m == 2
    yy, xx = y[fail], x[fail]
    if len(xx) == 0:
        return None
    fig = go.Figure(go.Scatter3d(
        x=xx, y=yy, z=np.ones(len(xx)),
        mode="markers",
        marker=dict(size=3.2, color=np.ones(len(xx)), colorscale=[[0,"#ff5577"],[1,"#ff5577"]], opacity=.82),
        hovertemplate="X %{x}<br>Y %{y}<extra>Observed failed die</extra>",
    ))
    fig.update_layout(
        title="Observed Failed-Die Spatial Distribution",
        scene=dict(
            xaxis=dict(backgroundcolor="#091526", gridcolor="#243d56", title="X"),
            yaxis=dict(backgroundcolor="#091526", gridcolor="#243d56", title="Y"),
            zaxis=dict(backgroundcolor="#091526", gridcolor="#243d56", title="Failure"),
            bgcolor="#091526",
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=45,b=0), height=430,
    )
    return fig


def kpi_card(label, value, sub=""):
    return (f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


def confusion_matrix_figure():
    candidates = [
        os.path.abspath("runs/eval_report_confusion.csv"),
        os.path.abspath("runs/eval_confusion.csv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                cm = pd.read_csv(path, index_col=0)
                fig = go.Figure(go.Heatmap(
                    z=cm.values,
                    x=[str(c).replace("pred ", "") for c in cm.columns],
                    y=[str(r).replace("true ", "") for r in cm.index],
                    colorscale=[[0, "#101d30"], [.5, "#477bff"], [1, "#ff5577"]],
                    hovertemplate="True %{y}<br>Predicted %{x}<br>Samples %{z}<extra></extra>",
                ))
                fig.update_layout(title="Actual Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")
                return plot_layout(fig, 380)
            except Exception:
                continue
    return None


@st.cache_data(show_spinner=False)
def load_wm811k_labeled(limit=1000):
    path = os.path.join(os.getcwd(), "data", "LSWMD_labelled.pkl")
    if not os.path.isfile(path):
        return None
    df = pd.read_pickle(path)
    if "waferMap" not in df.columns:
        return None
    if "failureType" in df.columns:
        df["label"] = df["failureType"].apply(
            lambda a: (a[0][0] if hasattr(a, "__len__") and len(a) else "none")
        )
    else:
        df["label"] = "none"
    df = df.head(limit).copy()
    maps = []
    names = []
    labels = []
    for i, row in df.iterrows():
        try:
            m = np.asarray(row["waferMap"])
            if m.ndim == 2:
                maps.append(resize_map(m, 64))
                names.append(str(row.get("waferIndex", i)))
                labels.append(str(row["label"]))
        except Exception:
            continue
    return {"maps": maps, "names": names, "labels": labels}


@st.cache_data(show_spinner=False)
def load_mixedwm38():
    path = os.path.join(os.getcwd(), "data", "Wafer_Map_Datasets.npz")
    if not os.path.isfile(path):
        return None
    d = np.load(path)
    if "arr_0" not in d.files:
        return None
    maps = d["arr_0"]
    maps = [resize_map(np.asarray(m),64) for m in maps[:1000]]
    return {"maps": maps, "names": [str(i) for i in range(len(maps))], "labels": ["Dataset label index unavailable"]*len(maps)}


st.markdown(
    '<header class="brand-header"><div class="logo-row"><div class="wafer-logo">WS</div>'
    '<h1 class="brand-title">WAFER SENTINEL</h1></div>'
    '<div class="brand-subtitle">AI-Powered Wafer Defect Detection, Root Cause Intelligence &amp; Yield Analytics</div>'
    '</header>', unsafe_allow_html=True)

cpts = sorted([os.path.abspath(os.path.join("runs", f)) for f in os.listdir("runs") if f.endswith(".pt")]) if os.path.isdir("runs") else []
ggufs = []
for root in ("models","data","."):
    if os.path.isdir(root):
        for dirpath, _, files in os.walk(root):
            if ".venv" not in dirpath:
                ggufs.extend(os.path.abspath(os.path.join(dirpath,f)) for f in files if f.lower().endswith(".gguf"))

st.markdown('<div class="status-chip good">SYSTEM ONLINE</div>'
            '<div class="status-chip warn">LOCAL ANALYSIS</div>'
            f'<div class="status-chip">{st.session_state.input_source or "NO INPUT SELECTED"}</div>', unsafe_allow_html=True)

tab_overview, tab_wafer, tab_root, tab_yield, tab_edge, tab_research, tab_status = st.tabs([
    "Overview","Wafer Intelligence","Root Cause","Yield Intelligence","Edge Monitor","Research Lab","System Status"
])

with tab_overview:
    left, center, right = st.columns([1,2.2,1])
    with left:
        st.markdown('<div class="glass-panel"><div class="panel-title">Analysis Configuration</div>', unsafe_allow_html=True)
        ckpt_choice = st.selectbox("CNN checkpoint",["None - use heuristic"]+cpts,
                                   key="cnn_checkpoint",
                                   format_func=lambda p: os.path.basename(p) if p != "None - use heuristic" else p)
        gguf_choice = st.selectbox("Llama GGUF",["None - rule-based"]+sorted(set(ggufs)),
                                   key="llama_gguf",
                                   format_func=lambda p: os.path.basename(p) if p != "None - rule-based" else p)
        baseline = st.slider("Baseline die yield reference",.80,.99,.95,.01,key="baseline_yield")
        if st.session_state.analysis is not None:
            if st.button("Clear current result",use_container_width=True):
                for key in ("analysis","map","processed_map","wafer_id","dataset_source","input_source","gradcam","uploaded_name","uploaded_bytes"):
                    st.session_state[key] = None
                st.success("Current wafer result cleared.")
                st.rerun()
        st.markdown('</div>',unsafe_allow_html=True)
    with center:
        st.markdown('<div class="glass-panel">'
                    '<div class="start-title">Start Wafer Inspection</div>'
                    '<div class="start-desc">Upload a wafer map or select a sample from the available datasets to begin AI-powered defect inspection.</div>'
                    '</div>',unsafe_allow_html=True)
        input_mode = st.radio("Input method",["Dataset sample","Synthetic sample","Upload .npy"],horizontal=True,key="input_method")
        pending_map = None
        pending_id = None
        pending_source = None

        if input_mode == "Dataset sample":
            dataset = st.selectbox("Dataset",["WM-811K","MixedWM38"],key="dataset_choice")
            if dataset == "WM-811K":
                dataset_data = load_wm811k_labeled()
            else:
                dataset_data = load_mixedwm38()
            if dataset_data is None:
                st.warning(f"{dataset} data is not available in the configured data directory.")
            else:
                sample_index = st.slider("Sample wafer",0,len(dataset_data["maps"])-1,0,key="dataset_sample")
                pending_map = dataset_data["maps"][sample_index]
                pending_id = dataset_data["names"][sample_index]
                pending_source = f"{dataset} / Wafer {pending_id}"
        elif input_mode == "Synthetic sample":
            pats = st.multiselect("Synthetic defect patterns",PATTERNS,["Edge-Ring"],key="synthetic_patterns")
            seed = st.number_input("Random seed",0,9999,42,key="synthetic_seed")
            size = st.select_slider("Wafer resolution",[32,48,64,96],64,key="synthetic_size")
            pending_map = make_map(pats,size,np.random.default_rng(int(seed)))
            pending_id = f"SYN-{int(seed):04d}-{size}"
            pending_source = "Synthetic"
        else:
            upload = st.file_uploader("Upload wafer map (.npy, 0/1/2 encoding)",type=["npy"],accept_multiple_files=False)
            if upload is not None:
                st.session_state.uploaded_name = upload.name
                st.session_state.uploaded_bytes = upload.getvalue()
            if st.session_state.uploaded_bytes is not None:
                try:
                    arr = np.load(io.BytesIO(st.session_state.uploaded_bytes),allow_pickle=False)
                    if arr.ndim != 2:
                        st.error("Unsupported file: expected a 2D NumPy wafer map.")
                    elif not set(np.unique(arr)).issubset({0,1,2}):
                        st.error("Unsupported encoding: values must be 0 (outside), 1 (good) or 2 (defective).")
                    else:
                        pending_map = arr.astype(np.uint8)
                        pending_id = os.path.splitext(st.session_state.uploaded_name)[0]
                        pending_source = f"Uploaded / {st.session_state.uploaded_name}"
                        st.success("Upload ready for inspection.")
                except Exception as exc:
                    st.error(f"Unable to read uploaded .npy file: {exc}")

        ckpt = None if ckpt_choice.startswith("None") else ckpt_choice
        gguf = None if gguf_choice.startswith("None") else gguf_choice
        if st.button("🔍  ANALYSE WAFER",type="primary",use_container_width=True):
            if pending_map is None:
                st.error("Select a valid dataset sample, synthetic configuration or supported .npy upload.")
            else:
                try:
                    with st.spinner("Running preprocessing and AI inference..."):
                        sentinel = Sentinel(ckpt,gguf)
                        result = sentinel.analyse(pending_map,baseline)
                        gradcam_map = None
                        if sentinel.model is not None and result["patterns"]:
                            import torch
                            from wafersentinel.data import to_tensor_input
                            class_idx = PATTERNS.index(result["patterns"][0])
                            gradcam_map = gradcam(sentinel.model,torch.tensor(to_tensor_input(pending_map[None])),class_idx)
                    st.session_state.analysis = result
                    st.session_state.map = pending_map
                    st.session_state.processed_map = pending_map
                    st.session_state.wafer_id = pending_id
                    st.session_state.dataset_source = dataset if input_mode == "Dataset sample" else pending_source
                    st.session_state.input_source = pending_source
                    st.session_state.gradcam = gradcam_map
                    st.session_state.history.insert(0,{
                        "timestamp":datetime.now().isoformat(timespec="seconds"),
                        "wafer_id":pending_id,
                        "dataset":st.session_state.dataset_source,
                        "patterns":", ".join(result["patterns"]) or "No systematic pattern",
                        "confidence":max(result["confidence"].values()) if result["confidence"] else 0,
                        "yield":result["stats"]["die_yield"],
                    })
                    st.success("Wafer analysis complete.")
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
    with right:
        st.markdown('<div class="glass-panel"><div class="panel-title">Model Status</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="status-chip {"good" if ckpt_choice != "None - use heuristic" else "warn"}">'
                    f'{"CNN CHECKPOINT" if ckpt_choice != "None - use heuristic" else "CNN HEURISTIC"}</div>'
                    f'<div class="status-chip {"good" if gguf_choice != "None - rule-based" else "warn"}">'
                    f'{"LLAMA GGUF" if gguf_choice != "None - rule-based" else "RULE ENGINE"}</div>'
                    '<div class="status-chip bad">HARDWARE NOT DETECTED</div></div>',unsafe_allow_html=True)

    if st.session_state.analysis is not None:
        result = st.session_state.analysis
        stats = result["stats"]
        patterns = result["patterns"]
        probs = result["probs"]
        primary = patterns[0] if patterns else "No systematic pattern"
        confidence = max(result["confidence"].values()) if result["confidence"] else 0.0
        low_conf = bool(patterns) and confidence < .6

        st.markdown('<div class="section-head"><h3>Inspection Results</h3>'
                    f'<span class="status-chip {"warn" if low_conf else "good"}">'
                    f'{"LOW CONFIDENCE – MANUAL REVIEW" if low_conf else "MODEL OUTPUT"}</span></div>',unsafe_allow_html=True)
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.markdown(kpi_card("Wafer",st.session_state.wafer_id or "Unavailable","Current sample"),unsafe_allow_html=True)
        c2.markdown(kpi_card("Predicted defect",primary,f"{confidence:.1%} confidence"),unsafe_allow_html=True)
        c3.markdown(kpi_card("Die yield",f"{stats['die_yield']:.1%}","Calculated estimate"),unsafe_allow_html=True)
        c4.markdown(kpi_card("Defect rate",f"{stats['failing_die']/max(stats['total_die'],1):.1%}","Observed on map"),unsafe_allow_html=True)
        c5.markdown(kpi_card("Total dies",stats['total_die'],"on map"),unsafe_allow_html=True)
        inference_ms = result.get("inference_ms")

        if inference_ms is None:
            inference_text = "N/A"
        else:
            inference_text = f"{inference_ms:.2f} ms"

        c6.markdown(
            kpi_card(
                "Inference",
                inference_text,
                "Measured local run"
            ),
            unsafe_allow_html=True
        )

        left,right = st.columns([1.55,1])
        with left:
            st.markdown('<div class="glass-panel"><div class="panel-title">Central Wafer Visualization</div>',unsafe_allow_html=True)
            st.plotly_chart(wafer_figure(st.session_state.map),use_container_width=True,config={"scrollZoom":True},key="overview-wafer")
            st.markdown('</div>',unsafe_allow_html=True)
        with right:
            st.markdown('<div class="glass-panel"><div class="panel-title">Prediction Summary</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="insight-card {"amber" if low_conf else "green"}"><b>Predicted category</b><br>'
                        f'{primary}<br><span class="small-note">{confidence:.1%} model confidence</span></div>',unsafe_allow_html=True)
            report = result["report"]
            st.markdown(f'<div class="insight-card"><b>Engineering insight</b><br><span class="small-note">'
                        f'{report["summary"]}</span></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="insight-card amber"><b>Yield observation</b><br><span class="small-note">'
                        f'{report["yield_impact"]}</span></div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        with c1:
            st.plotly_chart(bar_figure(probs,"Defect Distribution"),use_container_width=True,key="overview-distribution")
        with c2:
            st.plotly_chart(radial_figure(st.session_state.map),use_container_width=True,key="overview-radial")
        with c3:
            fig3d = defect_3d_figure(st.session_state.map)
            if fig3d:
                st.plotly_chart(fig3d,use_container_width=True,key="overview-3d")
            else:
                st.info("No observed failed dies are available for a 3D spatial view.")

        if st.session_state.gradcam is not None:
            cam = st.session_state.gradcam
            fig = go.Figure(go.Image(z=np.stack([np.clip(cam,0,1)*255]*3,axis=-1).astype(np.uint8)))
            fig.update_layout(title="Grad-CAM Explainability")
            st.plotly_chart(plot_layout(fig,420),use_container_width=True,key="overview-gradcam")

        die_df = pd.DataFrame([
            {"x":int(x),"y":int(y),"status":"Unavailable" if v==0 else ("Good" if v==1 else "Defective")}
            for y,row in enumerate(st.session_state.map) for x,v in enumerate(row)
        ])
        st.download_button("Export die-level CSV",die_df.to_csv(index=False).encode(),"wafer_die_analytics.csv","text/csv")

with tab_wafer:
    st.markdown('<div class="section-head"><h3>Wafer Intelligence</h3></div>',unsafe_allow_html=True)
    if st.session_state.analysis is None:
        st.info("Run an analysis on Overview to populate wafer intelligence.")
    else:
        result=st.session_state.analysis; m=st.session_state.map
        original_col, processed_col = st.columns(2)
        with original_col:
            st.markdown('<div class="panel-title">Original Wafer Map</div>',unsafe_allow_html=True)
            st.plotly_chart(wafer_figure(m,"Original 0/1/2 wafer map"),use_container_width=True,key="wafer-original")
        with processed_col:
            st.markdown('<div class="panel-title">Processed Model Input</div>',unsafe_allow_html=True)
            processed = st.session_state.processed_map if st.session_state.processed_map is not None else m
            st.plotly_chart(wafer_figure(processed,"Preprocessed categorical wafer map"),use_container_width=True,key="wafer-processed")
        mode=st.radio("Visualization",["Original wafer map","Defect distribution","Spatial heatmap","Classification overlay","Radial distribution"],horizontal=True)
        if mode=="Spatial heatmap":
            y,x=np.mgrid[0:m.shape[0],0:m.shape[1]]
            heat,_,_=np.histogram2d(y[m==2],x[m==2],bins=10,range=[[0,m.shape[0]],[0,m.shape[1]]])
            fig=go.Figure(go.Heatmap(z=heat,colorscale=[[0,"#101d30"],[.5,"#477bff"],[1,"#ff5577"]]))
            st.plotly_chart(plot_layout(fig.update_layout(title="Spatial Defect Heatmap"),470),use_container_width=True,key="wafer-heatmap")
        elif mode=="Radial distribution":
            st.plotly_chart(radial_figure(m),use_container_width=True,key="wafer-radial")
        else:
            st.plotly_chart(wafer_figure(m,mode),use_container_width=True,key="wafer-mode")
        c1,c2,c3,c4,c5=st.columns(5)
        stats=result["stats"]
        c1.metric("Total dies",stats["total_die"])
        c2.metric("Good dies",stats["total_die"]-stats["failing_die"])
        c3.metric("Defective dies",stats["failing_die"])
        c4.metric("Unavailable dies",int(np.count_nonzero(m==0)))
        c5.metric("Defect rate",f"{stats['failing_die']/max(stats['total_die'],1):.1%}")
        st.plotly_chart(bar_figure(result["probs"],"Multi-class prediction probabilities"),use_container_width=True,key="wafer-probabilities")

with tab_root:
    st.markdown('<div class="section-head"><h3>Root Cause Intelligence</h3></div>',unsafe_allow_html=True)
    if st.session_state.analysis is None:
        st.info("Run an analysis to generate root-cause hypotheses.")
    else:
        result=st.session_state.analysis; report=result["report"]
        backend = report.get("backend","Unavailable")
        is_llm = backend.startswith("llama")
        backend_label = "LOCAL LLAMA GGUF" if is_llm else ("RULE-BASED DOMAIN ENGINE" if backend.startswith("rule-based") else str(backend))
        confidence = max(result["confidence"].values()) if result["confidence"] else 0.0
        detected = html.escape(str(report.get("detected_pattern", ", ".join(result["patterns"]) or "No systematic pattern")))
        summary = html.escape(str(report.get("summary","No engineering summary is available.")))
        st.markdown(
            f'<div class="root-hero"><div class="status-chip {"good" if is_llm else "warn"}">{backend_label}</div>'
            f'<div class="status-chip">CNN confidence {confidence:.1%}</div>'
            f'<h3 style="margin:.65rem 0 .45rem;color:#f4f8ff;font-size:1.35rem;">Detected Pattern: {detected}</h3>'
            f'<p style="color:#b8c7da;line-height:1.65;margin:0;">{summary}</p></div>',
            unsafe_allow_html=True,
        )

        def report_list(items):
            if not items:
                return "<li>Not available from the current analysis.</li>"
            return "".join(f"<li>{html.escape(str(item))}</li>" for item in items)

        causes = report.get("likely_causes", [])
        evidence = report.get("supporting_evidence", [])
        processes = report.get("manufacturing_processes", [])
        actions = report.get("recommended_actions")
        if not actions:
            actions = [report.get("recommended_action","Continue engineering review.")]
        limitations = report.get("limitations", [])

        c1,c2=st.columns(2)
        with c1:
            st.markdown(
                f'<div class="root-grid-card"><h4>Root Cause Hypotheses</h4>'
                f'<ul>{report_list(causes)}</ul></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="root-grid-card"><h4>Supporting Evidence</h4>'
                f'<ul>{report_list(evidence)}</ul></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="root-grid-card"><h4>Manufacturing Process Involved</h4>'
                f'<ul>{report_list(processes)}</ul></div>',
                unsafe_allow_html=True,
            )
            yield_text = html.escape(str(report.get("yield_impact","Yield impact unavailable.")))
            st.markdown(
                f'<div class="root-grid-card"><h4>Potential Yield Impact</h4>'
                f'<p>{yield_text}</p></div>',
                unsafe_allow_html=True,
            )
        c1,c2=st.columns(2)
        with c1:
            st.markdown(
                f'<div class="root-grid-card"><h4>Recommended Actions</h4>'
                f'<ul>{report_list(actions)}</ul></div>',
                unsafe_allow_html=True,
            )
        with c2:
            confidence_text = html.escape(str(report.get("confidence_note","Confidence status unavailable.")))
            limitation_text = "<ul>" + report_list(limitations) + "</ul>"
            st.markdown(
                f'<div class="root-grid-card"><h4>Confidence &amp; Limitations</h4>'
                f'<p>{confidence_text}</p>{limitation_text}</div>',
                unsafe_allow_html=True,
            )

with tab_yield:
    st.markdown('<div class="section-head"><h3>Yield Intelligence</h3></div>',unsafe_allow_html=True)
    if st.session_state.analysis is None:
        st.info("Run an analysis to calculate wafer-map yield estimates.")
    else:
        result=st.session_state.analysis; stats=result["stats"]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Total dies",stats["total_die"])
        c2.metric("Good dies",stats["total_die"]-stats["failing_die"])
        c3.metric("Defective dies",stats["failing_die"])
        c4.metric("Estimated map yield",f"{stats['die_yield']:.1%}")
        c1,c2=st.columns(2)
        with c1:
            gauge=go.Figure(go.Indicator(mode="gauge+number",value=stats["die_yield"]*100,
                number={"suffix":"%"},gauge={"axis":{"range":[0,100]},"bar":{"color":"#32d9f5"},
                "steps":[{"range":[0,80],"color":"#3a2430"},{"range":[80,95],"color":"#25283a"},{"range":[95,100],"color":"#17382f"}]}))
            st.plotly_chart(plot_layout(gauge.update_layout(title="Calculated Die Yield"),360),use_container_width=True,key="yield-gauge")
        with c2:
            donut=go.Figure(go.Pie(labels=["Good","Defective"],values=[stats["total_die"]-stats["failing_die"],stats["failing_die"]],
                                   hole=.68,marker=dict(colors=["#31d69a","#ff5577"])))
            st.plotly_chart(plot_layout(donut.update_layout(title="Die Outcome"),360),use_container_width=True,key="yield-donut")
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(zone_figure(stats),use_container_width=True,key="yield-zone")
        with c2: st.plotly_chart(radial_figure(st.session_state.map),use_container_width=True,key="yield-radial")
        st.markdown('<div class="glass-panel"><div class="panel-title">Yield Methodology</div>'
                    '<span class="small-note">Die yield = good tested dies / total tested dies. This is an estimate from the 0/1/2 wafer-map encoding, not measured production yield. No cost or factory-loss claim is made.</span></div>',unsafe_allow_html=True)

with tab_edge:
    st.markdown('<div class="section-head"><h3>Edge Hardware Monitor</h3></div>',unsafe_allow_html=True)
    host=platform.node() or "Unavailable"; os_name=f"{platform.system()} {platform.release()}"; cpu_name=platform.processor() or "Unavailable"
    ram_total=ram_used=cpu_pct="Unavailable"
    if psutil:
        ram=psutil.virtual_memory(); ram_total=f"{ram.total/(1024**3):.2f} GB"; ram_used=f"{ram.percent:.1f}%"; cpu_pct=f"{psutil.cpu_percent(.2):.1f}%"
    hailo_status="NOT DETECTED"; hailo_info="No /dev/hailo0 device or hailortcli detection"
    if os.path.exists("/dev/hailo0"):
        hailo_status="DEVICE PRESENT"; hailo_info="/dev/hailo0 detected"
    else:
        try:
            probe=subprocess.run(["hailortcli","fw-control","identify"],capture_output=True,text=True,timeout=3)
            if probe.returncode==0:
                hailo_status="DETECTED"; hailo_info=probe.stdout[:500] or "Hailo CLI responded"
        except Exception: pass
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="glass-panel"><div class="panel-title">Host System</div>',unsafe_allow_html=True)
        for label,value in [("Host",host),("Operating system",os_name),("CPU",cpu_name),("Python",sys.version.split()[0]),("Available RAM",ram_total),("RAM utilization",ram_used),("CPU utilization",cpu_pct),("Hailo NPU",hailo_status),("Hailo information",hailo_info)]:
            st.markdown(f'<div class="insight-card"><b>{label}</b><br><span class="small-note">{value}</span></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-panel"><div class="panel-title">AI Model & Edge Status</div>',unsafe_allow_html=True)
        model_info=(st.session_state.analysis or {}).get("model",{"name":"Unavailable","format":"Unavailable","loaded":False,"input_size":"Unavailable","inference_device":"Unavailable"})
        model_file=ckpt_choice if ckpt_choice!="None - use heuristic" else None
        model_size=f"{os.path.getsize(model_file)/(1024**2):.2f} MB" if model_file and os.path.isfile(model_file) else "Unavailable"
        for label,value in [("Architecture","WaferCNN"),("Model format",model_info["format"].upper()),("Precision","FP32"),("Model size",model_size),("Input size",str(model_info["input_size"])),("Classes",len(PATTERNS)),("Inference device",model_info["inference_device"]),("Loading status","LOADED" if model_info["loaded"] else "HEURISTIC / NOT LOADED")]:
            st.markdown(f'<div class="insight-card"><b>{label}</b><br><span class="small-note">{value}</span></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    st.warning("Benchmark latency, throughput, memory acceleration and power values are unavailable unless measured on the target hardware.")

with tab_research:
    st.markdown('<div class="section-head"><h3>Research & Validation Lab</h3></div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="glass-panel"><div class="panel-title">Dataset Information</div>',unsafe_allow_html=True)
        wm=load_wm811k_labeled(); mixed=load_mixedwm38()
        for label,value in [("WM-811K","Available" if wm else "Not available"),("MixedWM38","Available" if mixed else "Not available"),("Synthetic generator","Available"),("Random seed","Configurable"),("Preprocessing","3-channel one-hot: outside / pass / fail"),("Dataset limitations","WM-811K lacks production yield and verified process-cause ground truth.")]:
            st.markdown(f'<div class="insight-card"><b>{label}</b><br><span class="small-note">{value}</span></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        if wm:
            labels=pd.Series(wm["labels"]).value_counts()
            st.plotly_chart(plot_layout(go.Figure(go.Bar(x=labels.index,y=labels.values,marker=dict(color="#477bff"))).update_layout(title="Loaded WM-811K Class Distribution"),380),use_container_width=True,key="research-class-distribution")
            rarest = labels.nsmallest(min(3,len(labels)))
            st.caption("Class imbalance - rarest loaded classes: " + ", ".join(f"{k} ({int(v)})" for k,v in rarest.items()))
        else:
            st.info("Dataset class distribution is unavailable until LSWMD_labelled.pkl is present.")

    eval_path=os.path.abspath("runs/eval_report.json")
    st.markdown('<div class="glass-panel"><div class="panel-title">CNN Evaluation</div>',unsafe_allow_html=True)
    if os.path.isfile(eval_path):
        try:
            ev=json.load(open(eval_path,"r",encoding="utf-8"))
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Macro F1",f"{ev.get('macro_f1',0):.3f}")
            c2.metric("Test wafers",ev.get("n_test","Unavailable"))
            c3.metric("False alarm",f"{ev.get('false_alarm_rate',0)*100:.2f}%")
            c4.metric("Miss rate",f"{ev.get('miss_rate',0)*100:.2f}%")
            pc=pd.DataFrame(ev.get("per_class",{})).T
            if not pc.empty:
                fig=go.Figure()
                for col,color in [("precision","#32d9f5"),("recall","#477bff"),("f1","#31d69a")]:
                    if col in pc: fig.add_trace(go.Bar(name=col.capitalize(),x=pc.index,y=pc[col],marker=dict(color=color)))
                st.plotly_chart(plot_layout(fig.update_layout(title="Per-class Performance"),380),use_container_width=True,key="research-per-class")
        except Exception as exc:
            st.error(f"Could not read evaluation report: {exc}")
    else:
        st.info("Actual training/validation curves and confusion-matrix metrics are unavailable. Generate runs/eval_report.json with evaluate.py.")
    cm_fig = confusion_matrix_figure()
    if cm_fig is not None:
        st.plotly_chart(cm_fig,use_container_width=True,key="research-confusion-matrix")
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="glass-panel"><div class="panel-title">Quantization Evaluation</div>',unsafe_allow_html=True)
    quant_candidates=[]
    export_dir=os.path.abspath("runs/export")
    if os.path.isdir(export_dir):
        quant_candidates=[os.path.join(export_dir,f) for f in os.listdir(export_dir) if f.endswith("_export.json")]
    if quant_candidates:
        st.json(json.load(open(quant_candidates[0],"r",encoding="utf-8")))
    else:
        st.info("FP32/INT8 comparison is unavailable. Generate export with wafersentinel.export.")
    st.markdown('</div>',unsafe_allow_html=True)

with tab_status:
    st.markdown('<div class="section-head"><h3>System Status</h3></div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    c1.markdown(kpi_card("Application","ONLINE","Streamlit local runtime"),unsafe_allow_html=True)
    model_loaded = False

if st.session_state.analysis:
    model_loaded = (
        st.session_state.analysis.get("model", {})
        .get("loaded", False)
    )

c2.markdown(
    kpi_card(
        "CNN",
        "CONFIGURED" if model_loaded else "HEURISTIC / NOT LOADED",
        "Checkpoint selection"
    ),
    unsafe_allow_html=True
)

c3.markdown(
    kpi_card(
        "Llama GGUF",
        "READY" if 'gguf' in locals() and gguf else "RULE-BASED",
        "Local reasoning status"
    ),
    unsafe_allow_html=True
)
if st.session_state.analysis:
    report_prediction = json.loads(json.dumps(st.session_state.analysis))
    if "impact" in report_prediction:
        report_prediction["impact"].pop("relative_cost_loss", None)
    payload={
        "wafer_id":st.session_state.wafer_id,
        "dataset_source":st.session_state.dataset_source,
        "input_source":st.session_state.input_source,
        "analysis_timestamp":datetime.now().isoformat(timespec="seconds"),
        "prediction":report_prediction,
        "methodology":"Die yield is calculated from the available 0/1/2 wafer map. The 95% baseline is a configurable reference, not measured production performance. Root causes are hypotheses.",
    }
    st.download_button("Download analysis report (JSON)",json.dumps(payload,indent=2).encode("utf-8"),
                       f"{st.session_state.wafer_id or 'wafer'}_analysis.json","application/json")
