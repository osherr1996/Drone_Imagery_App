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
HF_REPO_ID = "osherr/drone_app"
HF_REPO_TYPE = "dataset"

HF_TOKEN = st.secrets.get("HF_TOKEN", None)
if HF_TOKEN is None:
    HF_TOKEN = st.text_input("Enter Hugging Face token", type="password")

if not HF_TOKEN:
    st.warning("Please enter Hugging Face token.")
    st.stop()


# ============================================================
# BI BLOOM LEVEL SCALE
# ============================================================
BI_MIN = 0.0
BI_MAX = 5.0

BI_THRESHOLDS = {
    "Clean": 0.70,
    "Low": 1.07,
    "Medium": 1.35,
    "High": 1.70,
    "Very High": 2.50,
    "Extreme": 5.00,
}

# Clean -> Low -> Medium -> High -> Very High -> Extreme
BI_LEVEL_COLORS = np.array([
    [0,   110, 110],   # Clean
    [120, 190, 80],    # Low
    [255, 210, 0],     # Medium
    [255, 120, 70],    # High
    [220, 0,   80],    # Very High
    [100, 0,   0],     # Extreme
], dtype=np.float32)


# Severity palette remains separate, because severity is 0-3
SEVERITY_PALETTE = np.array([
    [0,   0,   139],
    [0,   160, 220],
    [0,   210, 210],
    [80,  200, 0],
    [255, 220, 0],
    [255, 140, 0],
    [120, 0,   0],
], dtype=np.float32)


def apply_continuous_palette(t_arr, alpha_arr, palette):
    n = len(palette) - 1
    idx = np.clip(t_arr * n, 0, n)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, n)
    frac = (idx - lo)[..., None]
    rgb = (palette[lo] * (1 - frac) + palette[hi] * frac).astype(np.uint8)
    return np.concatenate([rgb, alpha_arr[..., None]], axis=-1)


def bi_to_level_t(bi):
    """
    Convert BI to visual scale using fixed bloom-level thresholds:

    0.00-0.70  Clean
    0.70-1.07  Low
    1.07-1.35  Medium
    1.35-1.70  High
    1.70-2.50  Very High
    2.50-5.00  Extreme
    """
    bi = np.clip(bi, BI_MIN, BI_MAX)
    t = np.zeros_like(bi, dtype=np.float32)

    breaks = np.array([0.0, 0.70, 1.07, 1.35, 1.70, 2.50, 5.00], dtype=np.float32)
    visual = np.linspace(0.0, 1.0, len(breaks))

    for i in range(len(breaks) - 1):
        lo = breaks[i]
        hi = breaks[i + 1]
        m = (bi >= lo) & (bi <= hi)

        if hi > lo:
            local = (bi[m] - lo) / (hi - lo)
        else:
            local = 0

        t[m] = visual[i] + local * (visual[i + 1] - visual[i])

    return np.clip(t, 0, 1)


def bi_level_category(mean_bi):
    if mean_bi < 0.70:
        return "Clean"
    if mean_bi < 1.07:
        return "Low"
    if mean_bi < 1.35:
        return "Medium"
    if mean_bi < 1.70:
        return "High"
    if mean_bi < 2.50:
        return "Very High"
    return "Extreme"


# ============================================================
# HELPERS
# ============================================================
@st.cache_data(show_spinner=False)
def list_files(repo_id, repo_type, token):
    return list(list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token))


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
    m = re.search(r"_(\d{2}_\d{2}_\d{4})_", name)
    return m.group(1) if m else name


def date_sort_key(d):
    try:
        day, month, year = d.split("_")
        return int(year), int(month), int(day)
    except Exception:
        return 9999, 99, 99


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

    t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    lon_min, lat_min = t.transform(xmin, ymin)
    lon_max, lat_max = t.transform(xmax, ymax)

    return [[lat_min, lon_min], [lat_max, lon_max]]


def severity_category(mean_severity):
    if mean_severity < 1:
        return "Low"
    if mean_severity < 2:
        return "Medium"
    return "High"


def make_original_png(img_path):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)

    black = (
        (arr[:, :, 0] < 10) &
        (arr[:, :, 1] < 10) &
        (arr[:, :, 2] < 10)
    )

    rgba = np.zeros((*arr.shape[:2], 4), dtype=np.uint8)
    rgba[:, :, :3] = arr
    rgba[:, :, 3] = np.where(black, 0, 255).astype(np.uint8)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(rgba).save(tmp.name)

    return tmp.name


def make_severity_png(img_path):
    """
    Severity is still shown on its own 0-3 severity scale.
    """
    img = Image.open(img_path).convert("L")
    arr = np.array(img).astype(np.float32)

    valid = arr > 0
    severity = (arr / 255.0) * 3.0

    mean_sev = float(np.mean(severity[valid])) if valid.any() else 0.0

    t = np.clip(severity / 3.0, 0, 1)
    alpha = np.where(valid, 255, 0).astype(np.uint8)

    rgba = apply_continuous_palette(t, alpha, SEVERITY_PALETTE)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(rgba).save(tmp.name)

    return tmp.name, mean_sev


def make_pseudo_bi_png(img_path, bi_min=0.0, bi_max=5.0):
    """
    Pseudo-BI is colored using fixed BI bloom-level thresholds:

    Clean      0.00-0.70
    Low        0.70-1.07
    Medium     1.07-1.35
    High       1.35-1.70
    Very High  1.70-2.50
    Extreme    2.50-5.00
    """
    img = Image.open(img_path).convert("L")
    arr = np.array(img).astype(np.float32)

    valid = arr > 0

    bi = bi_min + (arr / 255.0) * (bi_max - bi_min)
    bi = np.clip(bi, BI_MIN, BI_MAX)

    t = bi_to_level_t(bi)
    alpha = np.where(valid, 255, 0).astype(np.uint8)

    rgba = apply_continuous_palette(t, alpha, BI_LEVEL_COLORS)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(rgba).save(tmp.name)

    mean_bi = float(np.mean(bi[valid])) if valid.any() else 0.0
    mean_bi_level = bi_level_category(mean_bi)

    return tmp.name, mean_bi, mean_bi_level


def make_high_bloom_prob_png(img_path):
    """
    Probability remains 0-1 probability scale.
    """
    img = Image.open(img_path).convert("L")
    arr = np.array(img).astype(np.float32)

    valid = arr > 0
    prob = arr / 255.0

    t = np.clip(prob, 0, 1)
    alpha = np.where(valid, 255, 0).astype(np.uint8)

    rgba = apply_continuous_palette(t, alpha, SEVERITY_PALETTE)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(rgba).save(tmp.name)

    mean_prob = float(np.mean(prob[valid])) if valid.any() else 0.0

    return tmp.name, mean_prob


def make_custom_legend():
    grad = (
        "linear-gradient(to top,"
        "rgb(0,110,110),"
        "rgb(120,190,80),"
        "rgb(255,210,0),"
        "rgb(255,120,70),"
        "rgb(220,0,80),"
        "rgb(100,0,0))"
    )

    return f"""
    <div style="position:fixed;bottom:30px;right:30px;z-index:9999;
    background:white;padding:12px;border:2px solid #444;border-radius:8px;
    font-size:13px;width:260px;box-shadow:0 2px 8px rgba(0,0,0,0.25);">

    <b>Bloom Index level</b>

    <div style="display:flex;gap:10px;margin-top:8px;">
        <div style="width:18px;height:180px;background:{grad};
        border:1px solid #333;border-radius:2px;"></div>

        <div style="font-size:12px;line-height:30px;">
            <div><b>Extreme</b> ≥ 2.50</div>
            <div><b>Very High</b> 1.70-2.50</div>
            <div><b>High</b> 1.35-1.70</div>
            <div><b>Medium</b> 1.07-1.35</div>
            <div><b>Low</b> 0.70-1.07</div>
            <div><b>Clean</b> 0.00-0.70</div>
        </div>
    </div>

    <hr style="margin:8px 0;"/>
    <div style="font-size:11px;color:#555;">
      BI layers use fixed bloom-index thresholds.<br>
      Severity and probability layers keep their own native scales.
    </div>
    </div>
    """


def add_layer_control_scroll(m, max_height="420px"):
    css = f"""
    <style>
    .leaflet-control-layers-expanded {{
        max-height: {max_height} !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-right: 8px !important;
    }}

    .leaflet-control-layers-list {{
        max-height: {max_height} !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }}

    .leaflet-control-layers-overlays label,
    .leaflet-control-layers-base label {{
        white-space: nowrap;
        font-size: 13px;
    }}
    </style>
    """

    m.get_root().header.add_child(folium.Element(css))


def make_timeline_chart(summary_df):
    spec = {
        "data": {"values": summary_df.to_dict("records")},
        "width": 360,
        "height": 220,
        "layer": [
            {
                "mark": {"type": "line", "color": "black", "opacity": 0.4},
                "encoding": {
                    "x": {
                        "field": "date",
                        "type": "ordinal",
                        "sort": None,
                        "axis": {"title": "Date"}
                    },
                    "y": {
                        "field": "mean_bi",
                        "type": "quantitative",
                        "scale": {"domain": [0, 5]},
                        "axis": {"title": "Mean Bloom Index"}
                    }
                }
            },
            {
                "mark": {"type": "circle", "size": 170},
                "encoding": {
                    "x": {
                        "field": "date",
                        "type": "ordinal",
                        "sort": None
                    },
                    "y": {
                        "field": "mean_bi",
                        "type": "quantitative",
                        "scale": {"domain": [0, 5]}
                    },
                    "color": {
                        "field": "mean_bi_level",
                        "type": "nominal",
                        "scale": {
                            "domain": [
                                "Clean",
                                "Low",
                                "Medium",
                                "High",
                                "Very High",
                                "Extreme"
                            ],
                            "range": [
                                "teal",
                                "yellowgreen",
                                "gold",
                                "coral",
                                "crimson",
                                "darkred"
                            ]
                        },
                        "legend": {"title": "BI level"}
                    },
                    "tooltip": [
                        {"field": "date", "type": "ordinal", "title": "Date"},
                        {"field": "mean_bi", "type": "quantitative", "title": "Mean BI", "format": ".2f"},
                        {"field": "mean_bi_level", "type": "nominal", "title": "BI Level"},
                        {"field": "severity_level", "type": "nominal", "title": "Severity Level"},
                        {
                            "field": "mean_high_bloom_prob",
                            "type": "quantitative",
                            "title": "High Bloom Prob",
                            "format": ".2f"
                        },
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
# FIND FILES
# ============================================================
def get_jpgs(prefix):
    return sorted([
        f for f in all_files
        if f.startswith(f"{selected_body}/{prefix}/")
        and f.lower().endswith(".jpg")
    ])


original_files = get_jpgs("original")
severity_files = get_jpgs("severity")
pseudo_bi_files = get_jpgs("pseudo_bi")
high_bloom_files = get_jpgs("high_bloom_prob")

if not severity_files:
    st.warning("No severity heatmaps found for this water body.")
    st.stop()


# ============================================================
# DOWNLOAD FILES
# ============================================================
def download_with_sidecars(hf_paths):
    local = {}

    for hf_path in hf_paths:
        date = extract_date_label(hf_path)

        local_jpg = download_file(
            HF_REPO_ID,
            hf_path,
            HF_REPO_TYPE,
            HF_TOKEN
        )

        for ext in [".jgw", ".prj"]:
            sidecar = os.path.splitext(hf_path)[0] + ext

            if sidecar in all_files:
                download_file(
                    HF_REPO_ID,
                    sidecar,
                    HF_REPO_TYPE,
                    HF_TOKEN
                )

        local[date] = local_jpg

    return local


with st.spinner("Downloading data..."):
    orig_by_date = download_with_sidecars(original_files)
    sev_by_date = download_with_sidecars(severity_files)
    bi_by_date = download_with_sidecars(pseudo_bi_files)
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

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=18,
    tiles=None
)

folium.TileLayer(
    "OpenStreetMap",
    name="OpenStreetMap",
    control=True
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Esri World Imagery",
    control=True,
).add_to(m)


summary_rows = []

for i, date in enumerate(all_dates):
    show_first = i == 0
    bounds = read_jgw_bounds(sev_by_date[date])

    if date in orig_by_date:
        ImageOverlay(
            name=f"{date} | Original",
            image=make_original_png(orig_by_date[date]),
            bounds=bounds,
            opacity=1.0,
            interactive=True,
            show=show_first,
        ).add_to(m)

    sev_png, mean_sev = make_severity_png(sev_by_date[date])

    ImageOverlay(
        name=f"{date} | Bloom Severity",
        image=sev_png,
        bounds=bounds,
        opacity=1.0,
        interactive=True,
        show=False,
    ).add_to(m)

    mean_bi = 0.0
    mean_bi_level = "Clean"

    if date in bi_by_date:
        bi_png, mean_bi, mean_bi_level = make_pseudo_bi_png(
            bi_by_date[date],
            bi_min=BI_MIN,
            bi_max=BI_MAX
        )

        ImageOverlay(
            name=f"{date} | Bloom Index",
            image=bi_png,
            bounds=bounds,
            opacity=1.0,
            interactive=True,
            show=False,
        ).add_to(m)

    mean_high_prob = 0.0

    if date in high_bloom_by_date:
        hp_png, mean_high_prob = make_high_bloom_prob_png(high_bloom_by_date[date])

        ImageOverlay(
            name=f"{date} | High Bloom Probability",
            image=hp_png,
            bounds=bounds,
            opacity=1.0,
            interactive=True,
            show=False,
        ).add_to(m)

    summary_rows.append({
        "date": date,
        "mean_severity": mean_sev,
        "severity_level": severity_category(mean_sev),
        "mean_bi": mean_bi,
        "mean_bi_level": mean_bi_level,
        "mean_high_bloom_prob": mean_high_prob,
    })


m.get_root().html.add_child(folium.Element(make_custom_legend()))

add_layer_control_scroll(m, max_height="420px")
folium.LayerControl(collapsed=False).add_to(m)


# ============================================================
# DISPLAY APP
# ============================================================
col1, col2 = st.columns([3, 1])

with col1:
    st.caption(
        "Use the layer control, top-right of map, to toggle: "
        "**Original**, **Bloom Severity**, **Bloom Index**, **High Bloom Probability**."
    )

    st_folium(m, width=1000, height=700)

with col2:
    st.subheader("Bloom Timeline")

    summary_df = pd.DataFrame(summary_rows)

    st.dataframe(
        summary_df[
            [
                "date",
                "mean_bi",
                "mean_bi_level",
                "severity_level",
                "mean_high_bloom_prob"
            ]
        ].rename(columns={
            "date": "Date",
            "mean_bi": "Mean BI",
            "mean_bi_level": "BI Level",
            "severity_level": "Severity Level",
            "mean_high_bloom_prob": "High Bloom Prob",
        }),
        hide_index=True,
        use_container_width=True,
    )

    make_timeline_chart(summary_df)

    st.metric("Latest BI Level", summary_df.iloc[-1]["mean_bi_level"])
    st.metric("Latest Mean BI", f"{summary_df.iloc[-1]['mean_bi']:.2f}")
    st.metric("Latest Severity Level", summary_df.iloc[-1]["severity_level"])
    st.metric(
        "Latest High Bloom Prob",
        f"{summary_df.iloc[-1]['mean_high_bloom_prob']:.2f}"
    )
