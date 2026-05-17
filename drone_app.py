import os
import re
import json
import tempfile
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import folium
from folium.raster_layers import ImageOverlay
from branca.colormap import LinearColormap
from pyproj import Transformer
from streamlit_folium import st_folium

from huggingface_hub import list_repo_files, hf_hub_download


# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Bloom Severity Viewer",
    layout="wide"
)

st.title("Bloom Severity Viewer")


# ============================================================
# HF CONFIG
# ============================================================
HF_REPO_ID = "osherr/drone_app"
HF_REPO_TYPE = "dataset"

# Put token in Streamlit secrets:
# HF_TOKEN = "hf_xxxxxxxxx"
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

if HF_TOKEN is None:
    HF_TOKEN = st.text_input(
        "Enter Hugging Face token",
        type="password"
    )

if not HF_TOKEN:
    st.warning("Please enter a Hugging Face token.")
    st.stop()


# ============================================================
# HELPERS
# ============================================================
@st.cache_data(show_spinner=False)
def list_files(repo_id, repo_type, token):
    return list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        token=token
    )


@st.cache_data(show_spinner=False)
def download_hf_file(repo_id, filename, repo_type, token):
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        token=token
    )


def extract_date_label(path):
    name = os.path.basename(path)
    m = re.search(r"_(\d{2}_\d{2})_", name)
    return m.group(1) if m else name


def read_jgw_bounds(img_path):
    jgw_path = os.path.splitext(img_path)[0] + ".jgw"

    with open(jgw_path, "r") as f:
        vals = [float(x.strip()) for x in f.readlines()]

    pixel_size_x = vals[0]
    pixel_size_y = vals[3]
    x_center = vals[4]
    y_center = vals[5]

    img = Image.open(img_path).convert("L")
    W, H = img.size

    xmin = x_center - pixel_size_x / 2
    ymax = y_center - pixel_size_y / 2
    xmax = xmin + pixel_size_x * W
    ymin = ymax + pixel_size_y * H

    transformer = Transformer.from_crs(
        "EPSG:3857",
        "EPSG:4326",
        always_xy=True
    )

    lon_min, lat_min = transformer.transform(xmin, ymin)
    lon_max, lat_max = transformer.transform(xmax, ymax)

    return [[lat_min, lon_min], [lat_max, lon_max]], img


def severity_category(mean_severity):
    if mean_severity < 1:
        return "Low"
    elif mean_severity < 2:
        return "Medium"
    return "High"


def make_colored_overlay(img):
    arr = np.array(img).astype(np.float32)
    severity = (arr / 255.0) * 3.0
    valid = arr > 0

    if np.sum(valid) > 0:
        mean_severity = float(np.mean(severity[valid]))
        median_severity = float(np.median(severity[valid]))
        max_severity = float(np.max(severity[valid]))
        water_pixel_count = int(np.sum(valid))
    else:
        mean_severity = 0.0
        median_severity = 0.0
        max_severity = 0.0
        water_pixel_count = 0

    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    s = np.clip(severity / 3.0, 0, 1)

    colors = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

    colors[..., 0] = np.clip(255 * np.maximum(0, 2 * s - 0.3), 0, 255)
    colors[..., 1] = np.clip(255 * (1 - np.abs(s - 0.45) * 1.8), 0, 255)
    colors[..., 2] = np.clip(255 * (1 - 2.2 * s), 0, 255)

    rgba[..., :3] = colors
    rgba[..., 3] = np.where(valid, 210, 0).astype(np.uint8)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(rgba).save(tmp.name)

    stats = {
        "date": None,
        "severity_level": severity_category(mean_severity),
        "mean_severity": mean_severity,
        "median_severity": median_severity,
        "max_severity": max_severity,
        "water_pixel_count": water_pixel_count,
    }

    return tmp.name, stats


def make_original_png(original_path):
    img = Image.open(original_path).convert("RGB")
    arr = np.array(img)

    black_mask = (
        (arr[:, :, 0] < 10) &
        (arr[:, :, 1] < 10) &
        (arr[:, :, 2] < 10)
    )

    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = arr
    rgba[:, :, 3] = np.where(black_mask, 0, 255).astype(np.uint8)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(rgba).save(tmp.name)

    return tmp.name


# ============================================================
# LOAD FILE LIST
# ============================================================
files = list_files(HF_REPO_ID, HF_REPO_TYPE, HF_TOKEN)

water_bodies = sorted({
    f.split("/")[0]
    for f in files
    if "/" in f
})

if not water_bodies:
    st.error("No water-body folders found in the dataset.")
    st.stop()

selected_body = st.selectbox(
    "Choose water body",
    water_bodies
)

heatmap_files = sorted([
    f for f in files
    if f.startswith(f"{selected_body}/heatmaps/")
    and f.lower().endswith(".jpg")
])

original_files = sorted([
    f for f in files
    if f.startswith(f"{selected_body}/original/")
    and f.lower().endswith(".jpg")
])

if len(heatmap_files) == 0:
    st.warning(f"No heatmaps found for {selected_body}.")
    st.stop()


# ============================================================
# DOWNLOAD REQUIRED FILES
# ============================================================
local_heatmaps = []
local_originals = []

for hf_path in heatmap_files:
    local_img = download_hf_file(HF_REPO_ID, hf_path, HF_REPO_TYPE, HF_TOKEN)
    local_jgw = download_hf_file(HF_REPO_ID, os.path.splitext(hf_path)[0] + ".jgw", HF_REPO_TYPE, HF_TOKEN)
    local_prj = download_hf_file(HF_REPO_ID, os.path.splitext(hf_path)[0] + ".prj", HF_REPO_TYPE, HF_TOKEN)
    local_heatmaps.append(local_img)

for hf_path in original_files:
    local_img = download_hf_file(HF_REPO_ID, hf_path, HF_REPO_TYPE, HF_TOKEN)
    local_originals.append(local_img)


# ============================================================
# MATCH ORIGINALS BY DATE
# ============================================================
original_by_date = {
    extract_date_label(p): p
    for p in local_originals
}

pairs = []

for h in local_heatmaps:
    d = extract_date_label(h)
    if d in original_by_date:
        pairs.append((h, original_by_date[d], d))

if len(pairs) == 0:
    st.error("Could not match heatmaps and originals by date.")
    st.stop()


# ============================================================
# CREATE MAP
# ============================================================
first_bounds, _ = read_jgw_bounds(pairs[0][0])

center_lat = (first_bounds[0][0] + first_bounds[1][0]) / 2
center_lon = (first_bounds[0][1] + first_bounds[1][1]) / 2

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=18,
    tiles=None
)

folium.TileLayer(
    tiles="OpenStreetMap",
    name="OpenStreetMap",
    control=True
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Esri World Imagery",
    control=True
).add_to(m)

summary_rows = []

for i, (heatmap_path, original_path, date_label) in enumerate(pairs):
    bounds, heat_img = read_jgw_bounds(heatmap_path)

    original_png = make_original_png(original_path)

    ImageOverlay(
        name=f"{date_label} | Original",
        image=original_png,
        bounds=bounds,
        opacity=1.0,
        interactive=True,
        show=(i == 0)
    ).add_to(m)

    heat_png, stats = make_colored_overlay(heat_img)
    stats["date"] = date_label

    ImageOverlay(
        name=f"{date_label} | Bloom severity: {stats['severity_level']}",
        image=heat_png,
        bounds=bounds,
        opacity=0.95,
        interactive=True,
        show=(i == 0)
    ).add_to(m)

    summary_rows.append(stats)

colormap = LinearColormap(
    colors=["blue", "green", "yellow", "orange", "red"],
    vmin=0,
    vmax=3,
    caption="Bloom severity: Low → Medium → High"
)

colormap.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)


# ============================================================
# SHOW APP
# ============================================================
col1, col2 = st.columns([3, 1])

with col1:
    st_folium(m, width=1000, height=650)

with col2:
    st.subheader("Severity timeline")

    summary_df = pd.DataFrame(summary_rows)

    display_df = summary_df[[
        "date",
        "severity_level",
        "water_pixel_count"
    ]].copy()

    display_df.columns = [
        "Date",
        "Bloom level",
        "Water pixels"
    ]

    st.dataframe(display_df, hide_index=True)

    st.line_chart(
        summary_df.set_index("date")["mean_severity"]
    )

    latest = summary_df.iloc[-1]["severity_level"]
    st.metric("Latest bloom level", latest)
