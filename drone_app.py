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
from huggingface_hub import (
    list_repo_files,
    hf_hub_download
)

import matplotlib.pyplot as plt


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

    m = re.search(r"_(\d{2}_\d{2})_", name)

    if m:
        return m.group(1)

    return name


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

    bounds = [
        [lat_min, lon_min],
        [lat_max, lon_max]
    ]

    return bounds, img


def severity_category(mean_severity):

    if mean_severity < 1:
        return "Low"

    elif mean_severity < 2:
        return "Medium"

    else:
        return "High"


def make_colored_overlay(img):

    arr = np.array(img).astype(np.float32)

    severity = (arr / 255.0) * 3.0

    valid = arr > 0

    if np.sum(valid) > 0:
        mean_severity = float(np.mean(severity[valid]))
    else:
        mean_severity = 0.0

    rgba = np.zeros(
        (arr.shape[0], arr.shape[1], 4),
        dtype=np.uint8
    )

    # ========================================================
    # LOW -> GREEN
    # MEDIUM -> ORANGE
    # HIGH -> RED
    # ========================================================
    low_mask = severity < 1
    med_mask = (severity >= 1) & (severity < 2)
    high_mask = severity >= 2

    rgba[low_mask] = [0, 255, 0, 190]
    rgba[med_mask] = [255, 165, 0, 190]
    rgba[high_mask] = [255, 0, 0, 190]

    rgba[~valid, 3] = 0

    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    )

    Image.fromarray(rgba).save(tmp.name)

    stats = {
        "severity_level": severity_category(mean_severity),
        "mean_severity": mean_severity
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

    rgba = np.zeros(
        (arr.shape[0], arr.shape[1], 4),
        dtype=np.uint8
    )

    rgba[:, :, :3] = arr

    rgba[:, :, 3] = np.where(
        black_mask,
        0,
        255
    ).astype(np.uint8)

    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    )

    Image.fromarray(rgba).save(tmp.name)

    return tmp.name


# ============================================================
# LOAD HF FILES
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
# GET FILES
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
# DOWNLOAD
# ============================================================
local_heatmaps = []
local_originals = []

with st.spinner("Downloading files..."):

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
# MATCH FILES BY DATE
# ============================================================
original_by_date = {
    extract_date_label(p): p
    for p in local_originals
}

pairs = []

for h in local_heatmaps:

    d = extract_date_label(h)

    if d in original_by_date:

        pairs.append(
            (
                h,
                original_by_date[d],
                d
            )
        )

pairs = sorted(
    pairs,
    key=date_sort_key
)

if len(pairs) == 0:
    st.error("Could not match files.")
    st.stop()


# ============================================================
# CREATE MAP
# ============================================================
first_bounds, _ = read_jgw_bounds(
    pairs[0][0]
)

center_lat = (
    first_bounds[0][0] +
    first_bounds[1][0]
) / 2

center_lon = (
    first_bounds[0][1] +
    first_bounds[1][1]
) / 2

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
# ADD LAYERS
# ============================================================
summary_rows = []

for i, (
    heatmap_path,
    original_path,
    date_label
) in enumerate(pairs):

    bounds, heat_img = read_jgw_bounds(
        heatmap_path
    )

    # ========================================================
    # ORIGINAL FIRST
    # ========================================================
    original_png = make_original_png(
        original_path
    )

    ImageOverlay(
        name=f"{date_label} | Original",
        image=original_png,
        bounds=bounds,
        opacity=1.0,
        interactive=True,
        show=(i == 0)
    ).add_to(m)

    # ========================================================
    # HEATMAP SECOND
    # ========================================================
    heat_png, stats = make_colored_overlay(
        heat_img
    )

    stats["date"] = date_label

    ImageOverlay(
        name=f"{date_label} | {stats['severity_level']} Bloom",
        image=heat_png,
        bounds=bounds,
        opacity=0.85,
        interactive=True,
        show=False
    ).add_to(m)

    summary_rows.append(stats)


# ============================================================
# CUSTOM LEGEND
# ============================================================
legend_html = """
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
">
<b>Bloom severity</b><br>
<span style="color:green;">■</span> Low<br>
<span style="color:orange;">■</span> Medium<br>
<span style="color:red;">■</span> High
</div>
"""

m.get_root().html.add_child(
    folium.Element(legend_html)
)

folium.LayerControl(
    collapsed=False
).add_to(m)


# ============================================================
# DISPLAY
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

    summary_df = pd.DataFrame(
        summary_rows
    )

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

    # ========================================================
    # PLOT
    # ========================================================
    level_to_value = {
        "Low": 1,
        "Medium": 2,
        "High": 3
    }

    level_to_color = {
        "Low": "green",
        "Medium": "orange",
        "High": "red"
    }

    summary_df["level_value"] = summary_df[
        "severity_level"
    ].map(level_to_value)

    fig, ax = plt.subplots(
        figsize=(5, 3)
    )

    for _, row in summary_df.iterrows():

        ax.scatter(
            row["date"],
            row["level_value"],
            s=180,
            color=level_to_color[
                row["severity_level"]
            ]
        )

    ax.plot(
        summary_df["date"],
        summary_df["level_value"],
        color="black",
        linewidth=1,
        alpha=0.4
    )

    ax.set_yticks([1, 2, 3])

    ax.set_yticklabels([
        "Low",
        "Medium",
        "High"
    ])

    ax.set_xlabel("Date")

    ax.set_ylabel("Bloom")

    ax.set_title(
        "Bloom Severity Timeline"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

    latest = summary_df.iloc[-1][
        "severity_level"
    ]

    st.metric(
        "Latest Bloom Level",
        latest
    )
