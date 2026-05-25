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
st.set_page_config(page_title="Relative Bloom Severity Viewer", layout="wide")
st.title("Reservoir Relative Bloom Severity Viewer")


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
# BLUE -> YELLOW -> RED PALETTE
# ============================================================
SEVERITY_PALETTE = np.array([
    [0,   80,  220],   # Low - blue
    [80,  170, 255],   # light blue
    [255, 235, 0],     # Low/Medium - yellow
    [255, 180, 0],     # orange-yellow
    [255, 90,  0],     # orange-red
    [220, 0,   0],     # red
    [120, 0,   0],     # dark red
], dtype=np.float32)


# ============================================================
# HELPERS
# ============================================================
def apply_continuous_palette(t_arr, alpha_arr, palette):
    n = len(palette) - 1
    idx = np.clip(t_arr * n, 0, n)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, n)
    frac = (idx - lo)[..., None]

    rgb = (palette[lo] * (1 - frac) + palette[hi] * frac).astype(np.uint8)
    return np.concatenate([rgb, alpha_arr[..., None]], axis=-1)


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
    if m:
        return m.group(1)

    m = re.search(r"(\d{2}_\d{2}_\d{4})", name)
    if m:
        return m.group(1)

    return name


def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format="%d_%m_%Y")
    except Exception:
        return pd.NaT


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

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    lon_min, lat_min = transformer.transform(xmin, ymin)
    lon_max, lat_max = transformer.transform(xmax, ymax)

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


def make_severity_legend():
    grad = (
        "linear-gradient(to top,"
        "rgb(0,80,220),"
        "rgb(255,235,0),"
        "rgb(220,0,0),"
        "rgb(120,0,0))"
    )

    return f"""
    <div style="position:fixed;bottom:30px;right:30px;z-index:9999;
    background:white;padding:12px;border:2px solid #444;border-radius:8px;
    font-size:13px;width:230px;box-shadow:0 2px 8px rgba(0,0,0,0.25);">

    <b>Relative Bloom Severity</b>

    <div style="display:flex;gap:10px;margin-top:8px;">
        <div style="width:18px;height:150px;background:{grad};
        border:1px solid #333;border-radius:2px;"></div>

        <div style="font-size:12px;line-height:50px;">
            <div><b>High</b></div>
            <div><b>Medium</b></div>
            <div><b>Low</b></div>
        </div>
    </div>

    <hr style="margin:8px 0;"/>
    <div style="font-size:11px;color:#555;">
      Severity is relative, scaled from 0 to 3.
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
    chart_df = summary_df.copy()
    chart_df["date_label"] = chart_df["date"]
    chart_df["date_sort"] = chart_df["date_dt"].dt.strftime("%Y-%m-%d")

    spec = {
        "data": {"values": chart_df.to_dict("records")},
        "width": 360,
        "height": 220,
        "layer": [
            {
                "mark": {"type": "line", "color": "black", "opacity": 0.4},
                "encoding": {
                    "x": {
                        "field": "date_label",
                        "type": "ordinal",
                        "sort": {"field": "date_sort", "order": "ascending"},
                        "axis": {"title": "Date"}
                    },
                    "y": {
                        "field": "mean_severity",
                        "type": "quantitative",
                        "scale": {"domain": [0, 3]},
                        "axis": {"title": "Mean Relative Severity"}
                    }
                }
            },
            {
                "mark": {"type": "circle", "size": 170},
                "encoding": {
                    "x": {
                        "field": "date_label",
                        "type": "ordinal",
                        "sort": {"field": "date_sort", "order": "ascending"}
                    },
                    "y": {
                        "field": "mean_severity",
                        "type": "quantitative",
                        "scale": {"domain": [0, 3]}
                    },
                    "color": {
                        "field": "severity_level",
                        "type": "nominal",
                        "scale": {
                            "domain": ["Low", "Medium", "High"],
                            "range": ["#0050dc", "#ffeb00", "#dc0000"]
                        },
                        "legend": {"title": "Severity"}
                    },
                    "tooltip": [
                        {"field": "date_label", "type": "ordinal", "title": "Date"},
                        {
                            "field": "mean_severity",
                            "type": "quantitative",
                            "title": "Mean Severity",
                            "format": ".2f"
                        },
                        {
                            "field": "severity_level",
                            "type": "nominal",
                            "title": "Severity Level"
                        },
                    ]
                }
            }
        ]
    }

    st.vega_lite_chart(spec, use_container_width=True)


# ============================================================
# LOAD FILE LIST
# ============================================================
all_files = list_files(HF_REPO_ID, HF_REPO_TYPE, HF_TOKEN)

water_bodies = sorted({f.split("/")[0] for f in all_files if "/" in f})

if not water_bodies:
    st.error("No water bodies found.")
    st.stop()

selected_body = st.selectbox("Choose water body", water_bodies)


def get_jpgs(prefix):
    return sorted([
        f for f in all_files
        if f.startswith(f"{selected_body}/{prefix}/")
        and f.lower().endswith(".jpg")
    ])


original_files = get_jpgs("original")
severity_files = get_jpgs("severity")

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


all_dates = sorted(
    sev_by_date.keys(),
    key=lambda d: parse_date(d) if pd.notna(parse_date(d)) else pd.Timestamp.max
)

if not all_dates:
    st.error("No dated files found.")
    st.stop()


# ============================================================
# BUILD MAP
# ============================================================
first_bounds = read_jgw_bounds(sev_by_date[all_dates[0]])

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
        name=f"{date} | Relative Severity",
        image=sev_png,
        bounds=bounds,
        opacity=1.0,
        interactive=True,
        show=False,
    ).add_to(m)

    summary_rows.append({
        "date": date,
        "date_dt": parse_date(date),
        "mean_severity": mean_sev,
        "severity_level": severity_category(mean_sev),
    })


m.get_root().html.add_child(folium.Element(make_severity_legend()))
add_layer_control_scroll(m, max_height="420px")
folium.LayerControl(collapsed=False).add_to(m)


# ============================================================
# DISPLAY APP
# ============================================================
col1, col2 = st.columns([3, 1])

with col1:
    st.caption(
        "Use the layer control, top-right of map, to toggle only "
        "**Original** and **Relative Severity**."
    )

    st_folium(m, width=1000, height=700)

with col2:
    st.subheader("Relative Severity Timeline")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("date_dt").reset_index(drop=True)

    st.dataframe(
        summary_df[
            [
                "date",
                "mean_severity",
                "severity_level",
            ]
        ].rename(columns={
            "date": "Date",
            "mean_severity": "Mean Severity",
            "severity_level": "Severity Level",
        }),
        hide_index=True,
        use_container_width=True,
    )

    make_timeline_chart(summary_df)

    latest = summary_df.iloc[-1]

    st.metric("Latest Severity Level", latest["severity_level"])
    st.metric("Latest Mean Severity", f"{latest['mean_severity']:.2f}")
