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
st.set_page_config(
    page_title="Bloom Severity Viewer",
    layout="wide"
)

st.title("Reservoir Bloom Severity Viewer")


# ============================================================
# HUGGING FACE CONFIG
# ============================================================
HF_REPO_ID = "osherr/drone_app"
HF_REPO_TYPE = "dataset"

HF_TOKEN = st.secrets.get("HF_TOKEN", None)

if HF_TOKEN is None:
    HF_TOKEN = st.text_input(
        "Enter Hugging Face token",
        type="password"
    )

if not HF_TOKEN:
    st.warning("Please enter Hugging Face token.")
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
def download_file(repo_id, filename, repo_type, token):
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        token=token
    )


def extract_date_label(path):
    name = os.path.basename(path)

    # matches _DD_MM_YYYY_
    m = re.search(r"_(\d{2}_\d{2}_\d{4})_", name)

    return m.group(1) if m else name


def date_sort_key(pair):
    d = pair[2]

    # split DD_MM_YYYY
    day, month, year = d.split("_")

    return int(year), int(month), int(day)


def date_sort_key(pair):
    d = pair[2]
    day, month = d.split("_")
    return int(month), int(day)


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


def make_continuous_colored_overlay(img):
    arr = np.array(img).astype(np.float32)

    severity = (arr / 255.0) * 3.0
    valid = arr > 0

    mean_severity = float(np.mean(severity[valid])) if np.sum(valid) > 0 else 0.0

    s = np.clip(severity / 3.0, 0, 1)

    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)

    # Continuous blue -> green -> yellow -> orange -> red
    colors = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

    colors[..., 0] = np.clip(
        255 * np.maximum(0, 2.2 * s - 0.25),
        0,
        255
    )

    colors[..., 1] = np.clip(
        255 * (1 - np.abs(s - 0.45) * 1.7),
        0,
        255
    )

    colors[..., 2] = np.clip(
        255 * (1 - 2.0 * s),
        0,
        255
    )

    rgba[..., :3] = colors
    rgba[..., 3] = np.where(valid, 210, 0).astype(np.uint8)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(rgba).save(tmp.name)

    stats = {
        "date": None,
        "mean_severity": mean_severity,
        "severity_level": severity_category(mean_severity)
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


def make_custom_legend():
    return """
    <div style="
    position: fixed;
    bottom: 30px;
    right: 30px;
    z-index: 9999;
    background: white;
    padding: 12px;
    border: 2px solid #444;
    border-radius: 8px;
    font-size: 14px;
    width: 210px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    ">
    <b>Bloom severity</b><br>
    <div style="
        height: 16px;
        margin-top: 8px;
        margin-bottom: 6px;
        background: linear-gradient(to right, blue, green, yellow, orange, red);
        border: 1px solid #555;
    "></div>
    <div style="display: flex; justify-content: space-between;">
        <span>Low</span>
        <span>Medium</span>
        <span>High</span>
    </div>
    </div>
    """


def make_timeline_chart(summary_df):
    chart_df = summary_df.copy()

    chart_df["Bloom level"] = chart_df["severity_level"]
    chart_df["Date"] = chart_df["date"]
    chart_df["Severity"] = chart_df["mean_severity"]

    spec = {
        "data": {
            "values": chart_df.to_dict("records")
        },
        "width": 360,
        "height": 250,
        "layer": [
            {
                "mark": {
                    "type": "line",
                    "point": False,
                    "color": "black",
                    "opacity": 0.45
                },
                "encoding": {
                    "x": {
                        "field": "Date",
                        "type": "ordinal",
                        "sort": None,
                        "axis": {
                            "title": "Date"
                        }
                    },
                    "y": {
                        "field": "Severity",
                        "type": "quantitative",
                        "scale": {
                            "domain": [0, 3]
                        },
                        "axis": {
                            "title": "Bloom level",
                            "values": [0.5, 1.5, 2.5],
                            "labelExpr": "datum.value < 1 ? 'Low' : datum.value < 2 ? 'Medium' : 'High'"
                        }
                    }
                }
            },
            {
                "mark": {
                    "type": "circle",
                    "size": 170
                },
                "encoding": {
                    "x": {
                        "field": "Date",
                        "type": "ordinal",
                        "sort": None
                    },
                    "y": {
                        "field": "Severity",
                        "type": "quantitative",
                        "scale": {
                            "domain": [0, 3]
                        }
                    },
                    "color": {
                        "field": "Bloom level",
                        "type": "nominal",
                        "scale": {
                            "domain": ["Low", "Medium", "High"],
                            "range": ["green", "orange", "red"]
                        },
                        "legend": {
                            "title": "Bloom level"
                        }
                    },
                    "tooltip": [
                        {"field": "Date", "type": "ordinal"},
                        {"field": "Bloom level", "type": "nominal"}
                    ]
                }
            }
        ]
    }

    st.vega_lite_chart(spec, use_container_width=True)


# ============================================================
# LOAD HF FILE LIST
# ============================================================
files = list_files(
    HF_REPO_ID,
    HF_REPO_TYPE,
    HF_TOKEN
)

water_bodies = sorted({
    f.split("/")[0]
    for f in files
    if "/" in f
})

if len(water_bodies) == 0:
    st.error("No water bodies found.")
    st.stop()


# ============================================================
# SELECT WATER BODY
# ============================================================
selected_body = st.selectbox(
    "Choose water body",
    water_bodies
)


# ============================================================
# FIND FILES
# ============================================================
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
    st.warning("No heatmaps found.")
    st.stop()


# ============================================================
# DOWNLOAD FILES
# ============================================================
local_heatmaps = []
local_originals = []

with st.spinner("Downloading data from private Hugging Face dataset..."):

    for hf_path in heatmap_files:
        local_img = download_file(
            HF_REPO_ID,
            hf_path,
            HF_REPO_TYPE,
            HF_TOKEN
        )

        download_file(
            HF_REPO_ID,
            os.path.splitext(hf_path)[0] + ".jgw",
            HF_REPO_TYPE,
            HF_TOKEN
        )

        download_file(
            HF_REPO_ID,
            os.path.splitext(hf_path)[0] + ".prj",
            HF_REPO_TYPE,
            HF_TOKEN
        )

        local_heatmaps.append(local_img)

    for hf_path in original_files:
        local_img = download_file(
            HF_REPO_ID,
            hf_path,
            HF_REPO_TYPE,
            HF_TOKEN
        )

        local_originals.append(local_img)


# ============================================================
# MATCH ORIGINALS AND HEATMAPS BY DATE
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

pairs = sorted(pairs, key=date_sort_key)

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


# ============================================================
# ADD LAYERS IN ORDER:
# original date1, heatmap date1, original date2, heatmap date2...
# ============================================================
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

    heat_png, stats = make_continuous_colored_overlay(heat_img)
    stats["date"] = date_label

    ImageOverlay(
        name=f"{date_label} | Heatmap | {stats['severity_level']} bloom",
        image=heat_png,
        bounds=bounds,
        opacity=0.90,
        interactive=True,
        show=False
    ).add_to(m)

    summary_rows.append(stats)


m.get_root().html.add_child(
    folium.Element(make_custom_legend())
)

folium.LayerControl(
    collapsed=False
).add_to(m)


# ============================================================
# DISPLAY APP
# ============================================================
col1, col2 = st.columns([3, 1])

with col1:
    st_folium(
        m,
        width=1000,
        height=700
    )

with col2:
    st.subheader("Bloom Timeline")

    summary_df = pd.DataFrame(summary_rows)

    st.dataframe(
        summary_df[[
            "date",
            "severity_level"
        ]].rename(columns={
            "date": "Date",
            "severity_level": "Bloom Level"
        }),
        hide_index=True
    )

    make_timeline_chart(summary_df)

    latest = summary_df.iloc[-1]["severity_level"]

    st.metric(
        "Latest Bloom Level",
        latest
    )
