import os
import re
import tempfile
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw, ImageFont

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
    page_title="HAB Bloom Viewer",
    layout="wide"
)

st.title("Reservoir HAB Bloom Viewer")


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
# COLOUR PALETTES
# ============================================================

# Severity: blue → yellow → red  (relative, 0–3)
SEVERITY_PALETTE = np.array([
    [0,   80,  220],
    [80,  170, 255],
    [255, 235,   0],
    [255, 180,   0],
    [255,  90,   0],
    [220,   0,   0],
    [120,   0,   0],
], dtype=np.float32)

# Pseudo-BI: dark teal → teal-green → lime → yellow → orange → red → dark red
# Matches the reference image: Clean(dark teal) → Low → Medium → High → Very High → Extreme
PSEUDO_BI_PALETTE = np.array([
    [0,   80,  60],    # dark teal   (Clean)
    [0,  130,  80],    # teal-green
    [80, 185,  50],    # lime green  (Low)
    [220, 220,  0],    # yellow      (Medium)
    [255, 140,  0],    # orange      (High)
    [200,  30,  0],    # red         (Very High)
    [100,   0,  0],    # dark red    (Extreme)
], dtype=np.float32)

BI_MIN = 0.6
BI_MAX = 5.0


# ============================================================
# HELPERS
# ============================================================
def apply_continuous_palette(t_arr, palette):
    n    = len(palette) - 1
    idx  = np.clip(t_arr * n, 0, n)
    lo   = np.floor(idx).astype(int)
    hi   = np.clip(lo + 1, 0, n)
    frac = (idx - lo)[..., None]
    return (palette[lo] * (1 - frac) + palette[hi] * frac).astype(np.uint8)


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
    with open(jgw_path) as f:
        vals = [float(x.strip()) for x in f.readlines()]
    pixel_size_x = vals[0]
    pixel_size_y = vals[3]
    x_center     = vals[4]
    y_center     = vals[5]
    img          = Image.open(img_path).convert("L")
    W, H         = img.size
    xmin = x_center - pixel_size_x / 2
    ymax = y_center - pixel_size_y / 2
    xmax = xmin + pixel_size_x * W
    ymin = ymax + pixel_size_y * H
    t    = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon_min, lat_min = t.transform(xmin, ymin)
    lon_max, lat_max = t.transform(xmax, ymax)
    return [[lat_min, lon_min], [lat_max, lon_max]]


def severity_category(mean_severity):
    if mean_severity < 1:  return "Low"
    if mean_severity < 2:  return "Medium"
    return "High"


def bi_category(mean_bi):
    if mean_bi < 1.5:  return "Clean"
    if mean_bi < 2.5:  return "Low bloom"
    if mean_bi < 3.5:  return "Medium bloom"
    return "High bloom"


def make_original_png(img_path):
    img  = Image.open(img_path).convert("RGB")
    arr  = np.array(img)
    black = (arr[:,:,0] < 10) & (arr[:,:,1] < 10) & (arr[:,:,2] < 10)
    rgba = np.zeros((*arr.shape[:2], 4), dtype=np.uint8)
    rgba[:,:,:3] = arr
    rgba[:,:,3]  = np.where(black, 0, 255).astype(np.uint8)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(rgba).save(tmp.name)
    return tmp.name


def make_severity_png(severity_img_path, original_img_path=None):
    """Severity heatmap blended over original. Returns (png_path, mean_severity)."""
    sev_img = Image.open(severity_img_path).convert("L")
    sev_arr = np.array(sev_img).astype(np.float32)
    valid   = sev_arr > 0
    severity = (sev_arr / 255.0) * 3.0
    mean_sev = float(np.mean(severity[valid])) if valid.any() else 0.0

    t        = np.clip(severity / 3.0, 0, 1)
    heat_rgb = apply_continuous_palette(t, SEVERITY_PALETTE)

    if original_img_path and os.path.exists(original_img_path):
        orig = Image.open(original_img_path).convert("RGB").resize(
            (sev_img.width, sev_img.height), Image.LANCZOS)
        orig_rgb = np.array(orig).astype(np.float32)
    else:
        orig_rgb = np.zeros_like(heat_rgb).astype(np.float32)

    blended = orig_rgb.copy()
    blended[valid] = 0.1 * orig_rgb[valid] + 0.9 * heat_rgb[valid]
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    alpha   = np.where(valid, 255, 0).astype(np.uint8)
    rgba    = np.dstack([blended, alpha])

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(rgba).save(tmp.name)
    return tmp.name, mean_sev


def make_pseudo_bi_png(pseudo_bi_img_path, original_img_path=None):
    """
    Pseudo-BI heatmap. The JPG is stored as grayscale where
    pixel=0 → BI_MIN and pixel=255 → BI_MAX.
    Returns (png_path, mean_bi).
    """
    bi_img  = Image.open(pseudo_bi_img_path).convert("L")
    bi_arr  = np.array(bi_img).astype(np.float32)
    valid   = bi_arr > 0

    # Rescale 0–255 → BI_MIN–BI_MAX
    bi_vals = bi_arr / 255.0 * (BI_MAX - BI_MIN) + BI_MIN
    mean_bi = float(np.mean(bi_vals[valid])) if valid.any() else BI_MIN

    t        = np.clip((bi_vals - BI_MIN) / (BI_MAX - BI_MIN), 0, 1)
    heat_rgb = apply_continuous_palette(t, PSEUDO_BI_PALETTE)

    if original_img_path and os.path.exists(original_img_path):
        orig = Image.open(original_img_path).convert("RGB").resize(
            (bi_img.width, bi_img.height), Image.LANCZOS)
        orig_rgb = np.array(orig).astype(np.float32)
    else:
        orig_rgb = np.zeros_like(heat_rgb).astype(np.float32)

    blended = orig_rgb.copy()
    blended[valid] = 0.1 * orig_rgb[valid] + 0.9 * heat_rgb[valid]
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    alpha   = np.where(valid, 255, 0).astype(np.uint8)
    rgba    = np.dstack([blended, alpha])

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(rgba).save(tmp.name)
    return tmp.name, mean_bi


def create_side_by_side_download(original_path, severity_path,
                                  pseudo_bi_path, date):
    panels = []
    labels = []

    if original_path and os.path.exists(original_path):
        panels.append(Image.open(original_path).convert("RGB"))
        labels.append("Original")

    if severity_path and os.path.exists(severity_path):
        sev_png, _ = make_severity_png(severity_path, original_path)
        panels.append(Image.open(sev_png).convert("RGB"))
        labels.append("Relative Severity")

    if pseudo_bi_path and os.path.exists(pseudo_bi_path):
        bi_png, _ = make_pseudo_bi_png(pseudo_bi_path, original_path)
        panels.append(Image.open(bi_png).convert("RGB"))
        labels.append("Pseudo Bloom Index")

    if not panels:
        return None

    target_h = min(min(p.height for p in panels), 900)

    def resize_keep_ratio(img, h):
        scale = h / img.height
        return img.resize((int(img.width * scale), h), Image.LANCZOS)

    panels = [resize_keep_ratio(p, target_h) for p in panels]
    title_h, gap, margin = 55, 20, 20
    out_w = sum(p.width for p in panels) + gap * (len(panels) - 1) + margin * 2
    out_h = target_h + title_h + margin * 2

    canvas = Image.new("RGB", (out_w, out_h), "white")
    draw   = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("arial.ttf", 26)
        font_label = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()

    draw.text((margin, 12),
              f"HAB Analysis | {date}",
              fill="black", font=font_title)

    x = margin
    y = margin + title_h
    for panel, label in zip(panels, labels):
        canvas.paste(panel, (x, y))
        draw.text((x, y - 25), label, fill="black", font=font_label)
        x += panel.width + gap

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    canvas.save(tmp.name)
    return tmp.name


def make_legend_html():
    # Severity: blue → yellow → red (no numbers, just Low/Medium/High)
    sev_grad = ("linear-gradient(to top,"
                "rgb(0,80,220),rgb(255,235,0),rgb(220,0,0))")
    # Pseudo-BI: dark teal → lime → yellow → orange → red → dark red
    bi_grad  = ("linear-gradient(to top,"
                "rgb(0,80,60),rgb(0,130,80),rgb(80,185,50),"
                "rgb(220,220,0),rgb(255,140,0),rgb(200,30,0),rgb(100,0,0))")
    return f"""
    <div style="position:fixed;bottom:30px;right:30px;z-index:9999;
    background:white;padding:12px;border:2px solid #444;border-radius:8px;
    font-size:13px;width:200px;box-shadow:0 2px 8px rgba(0,0,0,0.25);">

    <b>Relative Severity</b>
    <div style="display:flex;gap:10px;margin:6px 0 12px 0;">
        <div style="width:16px;height:90px;background:{sev_grad};
        border:1px solid #333;border-radius:2px;flex-shrink:0;"></div>
        <div style="font-size:11px;display:flex;flex-direction:column;
        justify-content:space-between;height:90px;">
            <div><b>High</b></div>
            <div><b>Medium</b></div>
            <div><b>Low</b></div>
        </div>
    </div>

    <b>Pseudo Bloom Index</b> &nbsp;<span style="font-size:11px;color:#555;">({BI_MIN}–{BI_MAX})</span>
    <div style="display:flex;gap:10px;margin:6px 0 6px 0;">
        <div style="width:16px;height:140px;background:{bi_grad};
        border:1px solid #333;border-radius:2px;flex-shrink:0;"></div>
        <div style="font-size:11px;display:flex;flex-direction:column;
        justify-content:space-between;height:140px;">
            <div><b>Extreme</b></div>
            <div><b>Very High</b></div>
            <div><b>High</b></div>
            <div><b>Medium</b></div>
            <div><b>Low</b></div>
            <div><b>Clean</b></div>
        </div>
    </div>

    <hr style="margin:6px 0;"/>
    <div style="font-size:10px;color:#666;">
      Severity: relative (SigLIP classes)<br>
      Pseudo-BI: calibrated to Sentinel-2 B5/B4
    </div>
    </div>"""


def add_layer_control_scroll(m, max_height="450px"):
    css = f"""<style>
    .leaflet-control-layers-expanded {{
        max-height:{max_height}!important;overflow-y:auto!important;
        overflow-x:hidden!important;padding-right:8px!important;}}
    .leaflet-control-layers-list {{
        max-height:{max_height}!important;overflow-y:auto!important;}}
    .leaflet-control-layers-overlays label,
    .leaflet-control-layers-base label {{
        white-space:nowrap;font-size:13px;}}
    </style>"""
    m.get_root().header.add_child(folium.Element(css))


def make_timeline_chart(summary_df):
    """Trendline showing Pseudo Bloom Index only."""
    chart_df = summary_df.copy()
    chart_df["date_sort"] = chart_df["date_dt"].dt.strftime("%Y-%m-%d")

    rows = []
    for _, r in chart_df.iterrows():
        if not np.isnan(r.get("mean_pseudo_bi", float("nan"))):
            rows.append({
                "date_label": r["date"],
                "date_sort":  r["date_sort"],
                "value":      r["mean_pseudo_bi"],
                "metric":     f"Pseudo-BI ({BI_MIN}–{BI_MAX})",
            })

    if not rows:
        st.info("No Pseudo-BI data available for the timeline.")
        return

    spec = {
        "data": {"values": rows},
        "width": 360, "height": 220,
        "layer": [
            {
                "mark": {"type": "line", "opacity": 0.7, "color": "#00a86b"},
                "encoding": {
                    "x": {"field": "date_label", "type": "ordinal",
                          "sort": {"field": "date_sort", "order": "ascending"},
                          "axis": {"title": "Date"}},
                    "y": {"field": "value", "type": "quantitative",
                          "axis": {"title": f"Pseudo-BI ({BI_MIN}–{BI_MAX})"},
                          "scale": {"domain": [BI_MIN, BI_MAX]}},
                }
            },
            {
                "mark": {"type": "circle", "size": 150, "color": "#00a86b"},
                "encoding": {
                    "x": {"field": "date_label", "type": "ordinal",
                          "sort": {"field": "date_sort", "order": "ascending"}},
                    "y": {"field": "value", "type": "quantitative"},
                    "tooltip": [
                        {"field": "date_label", "type": "ordinal",       "title": "Date"},
                        {"field": "value",      "type": "quantitative",  "title": "Pseudo-BI",
                         "format": ".3f"},
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


original_files   = get_jpgs("original")
severity_files   = get_jpgs("severity")
pseudo_bi_files  = get_jpgs("pseudo_bi")

if not severity_files:
    st.warning("No severity heatmaps found for this water body.")
    st.stop()


# ============================================================
# DOWNLOAD FILES
# ============================================================
def download_with_sidecars(hf_paths):
    local = {}
    for hf_path in hf_paths:
        date      = extract_date_label(hf_path)
        local_jpg = download_file(HF_REPO_ID, hf_path, HF_REPO_TYPE, HF_TOKEN)
        for ext in [".jgw", ".prj"]:
            sidecar = os.path.splitext(hf_path)[0] + ext
            if sidecar in all_files:
                download_file(HF_REPO_ID, sidecar, HF_REPO_TYPE, HF_TOKEN)
        local[date] = local_jpg
    return local


with st.spinner("Downloading data..."):
    orig_by_date      = download_with_sidecars(original_files)
    sev_by_date       = download_with_sidecars(severity_files)
    pseudo_bi_by_date = download_with_sidecars(pseudo_bi_files)

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
center_lat   = (first_bounds[0][0] + first_bounds[1][0]) / 2
center_lon   = (first_bounds[0][1] + first_bounds[1][1]) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles=None)

folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Esri World Imagery", control=True,
).add_to(m)

summary_rows = []

for i, date in enumerate(all_dates):
    show_first = (i == 0)
    bounds     = read_jgw_bounds(sev_by_date[date])
    orig_path  = orig_by_date.get(date)

    # ── Original ──────────────────────────────────────────────────────────
    if orig_path:
        ImageOverlay(
            name=f"{date} | Original",
            image=make_original_png(orig_path),
            bounds=bounds, opacity=1.0, interactive=True,
            show=show_first,
        ).add_to(m)

    # ── Severity ──────────────────────────────────────────────────────────
    sev_png, mean_sev = make_severity_png(sev_by_date[date], orig_path)
    ImageOverlay(
        name=f"{date} | Severity",
        image=sev_png,
        bounds=bounds, opacity=1.0, interactive=True,
        show=False,
    ).add_to(m)

    # ── Pseudo-BI ─────────────────────────────────────────────────────────
    mean_bi = float("nan")
    if date in pseudo_bi_by_date:
        bi_png, mean_bi = make_pseudo_bi_png(pseudo_bi_by_date[date], orig_path)
        ImageOverlay(
            name=f"{date} | Pseudo-BI",
            image=bi_png,
            bounds=bounds, opacity=1.0, interactive=True,
            show=False,
        ).add_to(m)

    summary_rows.append({
        "date":           date,
        "date_dt":        parse_date(date),
        "mean_severity":  mean_sev,
        "severity_level": severity_category(mean_sev),
        "mean_pseudo_bi": mean_bi,
        "bi_level":       bi_category(mean_bi) if not np.isnan(mean_bi) else "—",
    })

m.get_root().html.add_child(folium.Element(make_legend_html()))
add_layer_control_scroll(m, max_height="450px")
folium.LayerControl(collapsed=False).add_to(m)


# ============================================================
# DISPLAY APP
# ============================================================
col1, col2 = st.columns([3, 1])

with col1:
    st.caption("Use the layer control (top-right of map) to toggle layers.")
    st_folium(m, width=1000, height=700)

with col2:
    st.subheader("Timeline")

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("date_dt")
        .reset_index(drop=True)
    )

    display_cols = ["date", "mean_severity", "severity_level",
                    "mean_pseudo_bi", "bi_level"]
    rename_map   = {
        "date":           "Date",
        "mean_severity":  "Severity",
        "severity_level": "Sev. Level",
        "mean_pseudo_bi": "Pseudo-BI",
        "bi_level":       "BI Level",
    }

    st.dataframe(
        summary_df[display_cols].rename(columns=rename_map),
        hide_index=True,
        use_container_width=True,
    )

    make_timeline_chart(summary_df)

    latest = summary_df.iloc[-1]
    st.metric("Latest Severity Level", latest["severity_level"])
    st.metric("Latest Mean Severity",  f"{latest['mean_severity']:.2f}")

    if not np.isnan(latest["mean_pseudo_bi"]):
        st.metric("Latest BI Level",  latest["bi_level"])
        st.metric("Latest Pseudo-BI", f"{latest['mean_pseudo_bi']:.2f}")

    st.divider()
    st.subheader("Download Image")

    downloadable_dates = [
        d for d in all_dates
        if d in orig_by_date and d in sev_by_date
    ]

    if downloadable_dates:
        selected_dl = st.selectbox(
            "Choose date to download",
            downloadable_dates,
            index=len(downloadable_dates) - 1,
        )

        preview_path = create_side_by_side_download(
            orig_by_date.get(selected_dl),
            sev_by_date.get(selected_dl),
            pseudo_bi_by_date.get(selected_dl),
            selected_dl,
        )

        if preview_path:
            st.image(preview_path,
                     caption=f"Preview ({selected_dl})",
                     use_container_width=True)

            with open(preview_path, "rb") as f:
                st.download_button(
                    label="Download original + severity + pseudo-BI",
                    data=f,
                    file_name=(f"{selected_body}_{selected_dl}_analysis.png"),
                    mime="image/png",
                )
    else:
        st.info("No matching images found for download.")
