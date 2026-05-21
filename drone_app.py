import os
import re
import tempfile
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium

from pyproj import Transformer
from huggingface_hub import list_repo_files, hf_hub_download


# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(page_title="Bloom Severity Viewer", layout="wide")
st.title("Reservoir Bloom Severity Viewer")


# ============================================================
# HUGGING FACE CONFIG
# ============================================================
HF_REPO_ID   = "osherr/drone_app"
HF_REPO_TYPE = "dataset"

HF_TOKEN = st.secrets.get("HF_TOKEN", None)
if HF_TOKEN is None:
    HF_TOKEN = st.text_input("Enter Hugging Face token", type="password")
if not HF_TOKEN:
    st.warning("Please enter Hugging Face token.")
    st.stop()


# ============================================================
# HELPERS
# ============================================================
@st.cache_data(show_spinner=False)
def list_files(repo_id, repo_type, token):
    return list(list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token))


@st.cache_data(show_spinner=False)
def download_file(repo_id, filename, repo_type, token):
    return hf_hub_download(repo_id=repo_id, filename=filename,
                           repo_type=repo_type, token=token)


def extract_date_label(path):
    name = os.path.basename(path)
    m = re.search(r"_(\d{2}_\d{2}_\d{4})_", name)
    return m.group(1) if m else name


def date_sort_key(d):
    day, month, year = d.split("_")
    return int(year), int(month), int(day)


def read_jgw_bounds(img_path):
    jgw_path = os.path.splitext(img_path)[0] + ".jgw"
    with open(jgw_path, "r") as f:
        vals = [float(x.strip()) for x in f.readlines()]
    pixel_size_x = vals[0]
    pixel_size_y = vals[3]
    x_center     = vals[4]
    y_center     = vals[5]
    img = Image.open(img_path).convert("L")
    W, H = img.size
    xmin = x_center - pixel_size_x / 2
    ymax = y_center - pixel_size_y / 2
    xmax = xmin + pixel_size_x * W
    ymin = ymax + pixel_size_y * H
    t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon_min, lat_min = t.transform(xmin, ymin)
    lon_max, lat_max = t.transform(xmax, ymax)
    return [[lat_min, lon_min], [lat_max, lon_max]]


def severity_category(mean_severity):
    if mean_severity < 1: return "Low"
    if mean_severity < 2: return "Medium"
    return "High"


# ---- Original drone image -> transparent PNG ----
def make_original_png(img_path):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    black = (arr[:,:,0] < 10) & (arr[:,:,1] < 10) & (arr[:,:,2] < 10)
    rgba  = np.zeros((*arr.shape[:2], 4), dtype=np.uint8)
    rgba[:,:,:3] = arr
    rgba[:,:, 3] = np.where(black, 0, 255).astype(np.uint8)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(rgba).save(tmp.name)
    return tmp.name


# ---- Severity heatmap (grayscale 0-255 -> 0-3) -> blue/green/yellow/red RGBA ----
def make_severity_png(img_path):
    img = Image.open(img_path).convert("L")
    arr = np.array(img).astype(np.float32)
    severity = (arr / 255.0) * 3.0
    valid    = arr > 0
    mean_sev = float(np.mean(severity[valid])) if valid.any() else 0.0
    s = np.clip(severity / 3.0, 0, 1)
    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    rgba[:,:,0] = np.clip(255 * np.maximum(0, 2.2*s - 0.25), 0, 255)
    rgba[:,:,1] = np.clip(255 * (1 - np.abs(s - 0.45)*1.7),  0, 255)
    rgba[:,:,2] = np.clip(255 * (1 - 2.0*s),                  0, 255)
    rgba[:,:,3] = np.where(valid, 210, 0).astype(np.uint8)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(rgba).save(tmp.name)
    return tmp.name, mean_sev


# ---- Pseudo-BI heatmap (grayscale 0-255 -> BI_MIN-BI_MAX) -> viridis RGBA ----
def make_pseudo_bi_png(img_path, bi_min=0.6, bi_max=3.5):
    img   = Image.open(img_path).convert("L")
    arr   = np.array(img).astype(np.float32)
    valid = arr > 0
    bi    = bi_min + (arr / 255.0) * (bi_max - bi_min)
    t     = np.clip((bi - bi_min) / (bi_max - bi_min), 0, 1)
    rgba  = np.zeros((*arr.shape, 4), dtype=np.uint8)
    rgba[:,:,0] = np.clip(255 * (0.28 + 0.85*t - 0.5*t**2),  0, 255)
    rgba[:,:,1] = np.clip(255 * (0.00 + 1.20*t - 0.20*t**2), 0, 255)
    rgba[:,:,2] = np.clip(255 * (0.50 - 0.90*t + 0.40*t**2), 0, 255)
    rgba[:,:,3] = np.where(valid, 210, 0).astype(np.uint8)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(rgba).save(tmp.name)
    mean_bi = float(np.mean(bi[valid])) if valid.any() else 0.0
    return tmp.name, mean_bi


# ---- High bloom probability (grayscale 0-255 -> 0-1) -> white-to-red RGBA ----
def make_high_bloom_prob_png(img_path):
    img   = Image.open(img_path).convert("L")
    arr   = np.array(img).astype(np.float32)
    valid = arr > 0
    prob  = arr / 255.0
    rgba  = np.zeros((*arr.shape, 4), dtype=np.uint8)
    rgba[:,:,0] = 255
    rgba[:,:,1] = np.clip(255 * (1 - prob), 0, 255).astype(np.uint8)
    rgba[:,:,2] = np.clip(255 * (1 - prob), 0, 255).astype(np.uint8)
    rgba[:,:,3] = np.where(valid, 210, 0).astype(np.uint8)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(rgba).save(tmp.name)
    mean_prob = float(np.mean(prob[valid])) if valid.any() else 0.0
    return tmp.name, mean_prob


def make_custom_legend():
    return """
    <div style="position:fixed;bottom:30px;right:30px;z-index:9999;
    background:white;padding:12px;border:2px solid #444;border-radius:8px;
    font-size:13px;width:220px;box-shadow:0 2px 8px rgba(0,0,0,0.25);">
    <b>Bloom Severity</b>
    <div style="height:14px;margin:6px 0 4px;
    background:linear-gradient(to right,blue,green,yellow,orange,red);
    border:1px solid #555;"></div>
    <div style="display:flex;justify-content:space-between;">
        <span>Low</span><span>Medium</span><span>High</span>
    </div>
    <hr style="margin:8px 0;">
    <b>Pseudo-BI</b>
    <div style="height:14px;margin:6px 0 4px;
    background:linear-gradient(to right,#440154,#3b528b,#21918c,#5ec962,#fde725);
    border:1px solid #555;"></div>
    <div style="display:flex;justify-content:space-between;">
        <span>Low</span><span>High</span>
    </div>
    <hr style="margin:8px 0;">
    <b>High Bloom Probability</b>
    <div style="height:14px;margin:6px 0 4px;
    background:linear-gradient(to right,white,pink,red);
    border:1px solid #555;"></div>
    <div style="display:flex;justify-content:space-between;">
        <span>0</span><span>1</span>
    </div>
    </div>
    """


def make_timeline_chart(summary_df):
    spec = {
        "data": {"values": summary_df.to_dict("records")},
        "width": 360, "height": 220,
        "layer": [
            {
                "mark": {"type": "line", "color": "black", "opacity": 0.4},
                "encoding": {
                    "x": {"field": "date", "type": "ordinal", "sort": None,
                          "axis": {"title": "Date"}},
                    "y": {"field": "mean_severity", "type": "quantitative",
                          "scale": {"domain": [0, 3]},
                          "axis": {"title": "Bloom level", "values": [0.5,1.5,2.5],
                                   "labelExpr": "datum.value<1?'Low':datum.value<2?'Medium':'High'"}}
                }
            },
            {
                "mark": {"type": "circle", "size": 170},
                "encoding": {
                    "x": {"field": "date", "type": "ordinal", "sort": None},
                    "y": {"field": "mean_severity", "type": "quantitative",
                          "scale": {"domain": [0, 3]}},
                    "color": {
                        "field": "severity_level", "type": "nominal",
                        "scale": {"domain": ["Low","Medium","High"],
                                  "range": ["green","orange","red"]},
                        "legend": {"title": "Bloom level"}
                    },
                    "tooltip": [
                        {"field": "date", "type": "ordinal", "title": "Date"},
                        {"field": "severity_level", "type": "nominal", "title": "Level"},
                        {"field": "mean_bi", "type": "quantitative", "title": "Mean BI",
                         "format": ".2f"},
                        {"field": "mean_high_bloom_prob", "type": "quantitative",
                         "title": "High Bloom Prob", "format": ".2f"},
                    ]
                }
            }
        ]
    }
    st.vega_lite_chart(spec, use_container_width=True)


# ============================================================
# LOAD HF FILE LIST
# ============================================================
all_files = list_files(HF_REPO_ID, HF_REPO_TYPE, HF_TOKEN)

water_bodies = sorted({f.split("/")[0] for f in all_files if "/" in f})
if not water_bodies:
    st.error("No water bodies found.")
    st.stop()

selected_body = st.selectbox("Choose water body", water_bodies)


# ============================================================
# FIND FILES FOR SELECTED WATER BODY
# ============================================================
def get_jpgs(prefix):
    return sorted([f for f in all_files
                   if f.startswith(f"{selected_body}/{prefix}/")
                   and f.lower().endswith(".jpg")])

original_files       = get_jpgs("original")
severity_files       = get_jpgs("severity")
pseudo_bi_files      = get_jpgs("pseudo_bi")
high_bloom_files     = get_jpgs("high_bloom_prob")   # <-- high bloom only

if not severity_files:
    st.warning("No severity heatmaps found for this water body.")
    st.stop()


# ============================================================
# DOWNLOAD ALL FILES
# ============================================================
def download_with_sidecars(hf_paths):
    local = {}
    for hf_path in hf_paths:
        date = extract_date_label(hf_path)
        local_jpg = download_file(HF_REPO_ID, hf_path, HF_REPO_TYPE, HF_TOKEN)
        for ext in [".jgw", ".prj"]:
            sidecar = os.path.splitext(hf_path)[0] + ext
            if sidecar in all_files:
                download_file(HF_REPO_ID, sidecar, HF_REPO_TYPE, HF_TOKEN)
        local[date] = local_jpg
    return local

with st.spinner("Downloading data..."):
    orig_by_date       = download_with_sidecars(original_files)
    sev_by_date        = download_with_sidecars(severity_files)
    bi_by_date         = download_with_sidecars(pseudo_bi_files)
    high_bloom_by_date = download_with_sidecars(high_bloom_files)

all_dates = sorted(sev_by_date.keys(), key=date_sort_key)

if not all_dates:
    st.error("No dated files found.")
    st.stop()


# ============================================================
# BUILD MAP
# ============================================================
first_bounds = read_jgw_bounds(list(sev_by_date.values())[0])
center_lat = (first_bounds[0][0] + first_bounds[1][0]) / 2
center_lon = (first_bounds[0][1] + first_bounds[1][1]) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles=None)

folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Esri World Imagery", control=True,
).add_to(m)

summary_rows = []

for i, date in enumerate(all_dates):
    show_first = (i == 0)
    bounds = read_jgw_bounds(sev_by_date[date])

    # -- Original --
    if date in orig_by_date:
        ImageOverlay(
            name=f"{date} | Original",
            image=make_original_png(orig_by_date[date]),
            bounds=bounds, opacity=1.0, interactive=True, show=show_first,
        ).add_to(m)

    # -- Bloom Severity --
    sev_png, mean_sev = make_severity_png(sev_by_date[date])
    ImageOverlay(
        name=f"{date} | Bloom Severity",
        image=sev_png,
        bounds=bounds, opacity=0.90, interactive=True, show=False,
    ).add_to(m)

    # -- Pseudo-BI --
    mean_bi = 0.0
    if date in bi_by_date:
        bi_png, mean_bi = make_pseudo_bi_png(bi_by_date[date])
        ImageOverlay(
            name=f"{date} | Pseudo-BI",
            image=bi_png,
            bounds=bounds, opacity=0.90, interactive=True, show=False,
        ).add_to(m)

    # -- High Bloom Probability --
    mean_high_prob = 0.0
    if date in high_bloom_by_date:
        hp_png, mean_high_prob = make_high_bloom_prob_png(high_bloom_by_date[date])
        ImageOverlay(
            name=f"{date} | High Bloom Probability",
            image=hp_png,
            bounds=bounds, opacity=0.90, interactive=True, show=False,
        ).add_to(m)

    summary_rows.append({
        "date": date,
        "mean_severity": mean_sev,
        "severity_level": severity_category(mean_sev),
        "mean_bi": mean_bi,
        "mean_high_bloom_prob": mean_high_prob,
    })

m.get_root().html.add_child(folium.Element(make_custom_legend()))
folium.LayerControl(collapsed=False).add_to(m)


# ============================================================
# DISPLAY APP
# ============================================================
col1, col2 = st.columns([3, 1])

with col1:
    st.caption("Use the layer control (top-right of map) to toggle: "
               "**Original**, **Bloom Severity**, **Pseudo-BI**, **High Bloom Probability**.")
    st_folium(m, width=1000, height=700)

with col2:
    st.subheader("Bloom Timeline")
    summary_df = pd.DataFrame(summary_rows)

    st.dataframe(
        summary_df[["date","severity_level","mean_bi","mean_high_bloom_prob"]].rename(columns={
            "date": "Date",
            "severity_level": "Level",
            "mean_bi": "Mean BI",
            "mean_high_bloom_prob": "High Bloom Prob",
        }),
        hide_index=True,
    )

    make_timeline_chart(summary_df)

    st.metric("Latest Bloom Level",      summary_df.iloc[-1]["severity_level"])
    st.metric("Latest Mean BI",          f"{summary_df.iloc[-1]['mean_bi']:.2f}")
    st.metric("Latest High Bloom Prob",  f"{summary_df.iloc[-1]['mean_high_bloom_prob']:.2f}")
