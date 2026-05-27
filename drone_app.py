
import os
import re
import tempfile
import subprocess
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
st.set_page_config(page_title="HAB Bloom Viewer", layout="wide")
st.title("Reservoir HAB Bloom Viewer")


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
# PALETTES
# ============================================================
SEVERITY_PALETTE = np.array([
    [0, 80, 220],
    [80, 170, 255],
    [255, 235, 0],
    [255, 180, 0],
    [255, 90, 0],
    [220, 0, 0],
    [120, 0, 0],
], dtype=np.float32)

PSEUDO_BI_PALETTE = np.array([
    [0, 80, 60],
    [0, 130, 80],
    [80, 185, 50],
    [220, 220, 0],
    [255, 140, 0],
    [200, 30, 0],
    [100, 0, 0],
], dtype=np.float32)

BI_MIN = 0.6
BI_MAX = 5.0


# ============================================================
# BASIC HELPERS
# ============================================================
def apply_continuous_palette(t_arr, palette):
    n = len(palette) - 1
    idx = np.clip(t_arr * n, 0, n)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, n)
    frac = (idx - lo)[..., None]
    return (palette[lo] * (1 - frac) + palette[hi] * frac).astype(np.uint8)


@st.cache_data(show_spinner=False)
def list_files(repo_id, repo_type, token):
    return list(list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token))


@st.cache_data(show_spinner=False)
def download_file(repo_id, filename, repo_type, token):
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        token=token,
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


def normalize_name_for_matching(name):
    out = name.lower()
    for token in [
        "_warped_to_satellite_georef_final_severity_index_georef",
        "_warped_to_satellite_georef_final_severity_index_georef_severity",
        "_severity",
        "_pseudo_bi",
        "_pseudobi",
        "_bi",
        "_original",
    ]:
        out = out.replace(token, "")
    out = re.sub(r"[^a-z0-9]+", "_", out)
    out = re.sub(r"_+", "_", out).strip("_")
    return out


def read_jgw_bounds(img_path):
    jgw_path = os.path.splitext(img_path)[0] + ".jgw"
    with open(jgw_path) as f:
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


def bi_category(mean_bi):
    if mean_bi < 1.5:
        return "Clean"
    if mean_bi < 2.5:
        return "Low bloom"
    if mean_bi < 3.5:
        return "Medium bloom"
    return "High bloom"


def safe_font(size=80, bold=False):

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",

        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold else
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",

        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]

    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass

    return ImageFont.load_default()



# ============================================================
# IMAGE CREATION
# ============================================================
def make_original_png(img_path):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)

    black = (arr[:, :, 0] < 10) & (arr[:, :, 1] < 10) & (arr[:, :, 2] < 10)

    rgba = np.zeros((*arr.shape[:2], 4), dtype=np.uint8)
    rgba[:, :, :3] = arr
    rgba[:, :, 3] = np.where(black, 0, 255).astype(np.uint8)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(rgba).save(tmp.name)
    return tmp.name


def make_severity_png(severity_img_path, original_img_path=None):
    sev_img = Image.open(severity_img_path).convert("L")
    sev_arr = np.array(sev_img).astype(np.float32)

    valid = sev_arr > 0
    severity = (sev_arr / 255.0) * 3.0
    mean_sev = float(np.mean(severity[valid])) if valid.any() else 0.0

    t = np.clip(severity / 3.0, 0, 1)
    heat_rgb = apply_continuous_palette(t, SEVERITY_PALETTE)

    if original_img_path and os.path.exists(original_img_path):
        orig = Image.open(original_img_path).convert("RGB").resize(
            (sev_img.width, sev_img.height), Image.LANCZOS
        )
        orig_rgb = np.array(orig).astype(np.float32)
    else:
        orig_rgb = np.zeros_like(heat_rgb).astype(np.float32)

    blended = orig_rgb.copy()
    blended[valid] = 0.1 * orig_rgb[valid] + 0.9 * heat_rgb[valid]
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    alpha = np.where(valid, 255, 0).astype(np.uint8)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(np.dstack([blended, alpha])).save(tmp.name)
    return tmp.name, mean_sev


def make_pseudo_bi_png(pseudo_bi_img_path, original_img_path=None):
    bi_img = Image.open(pseudo_bi_img_path).convert("L")
    bi_arr = np.array(bi_img).astype(np.float32)

    valid = bi_arr > 0
    bi_vals = bi_arr / 255.0 * (BI_MAX - BI_MIN) + BI_MIN
    mean_bi = float(np.mean(bi_vals[valid])) if valid.any() else BI_MIN

    t = np.clip((bi_vals - BI_MIN) / (BI_MAX - BI_MIN), 0, 1)
    heat_rgb = apply_continuous_palette(t, PSEUDO_BI_PALETTE)

    if original_img_path and os.path.exists(original_img_path):
        orig = Image.open(original_img_path).convert("RGB").resize(
            (bi_img.width, bi_img.height), Image.LANCZOS
        )
        orig_rgb = np.array(orig).astype(np.float32)
    else:
        orig_rgb = np.zeros_like(heat_rgb).astype(np.float32)

    blended = orig_rgb.copy()
    blended[valid] = 0.1 * orig_rgb[valid] + 0.9 * heat_rgb[valid]
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    alpha = np.where(valid, 255, 0).astype(np.uint8)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(np.dstack([blended, alpha])).save(tmp.name)
    return tmp.name, mean_bi


def add_title_to_canvas(draw, text, x, y, font, fill="black"):
    draw.text((x, y), text, fill=fill, font=font)


def create_side_by_side_download(
    original_path,
    severity_path=None,
    pseudo_bi_path=None,
    date="",
    location="",
    display_name="",
    mode="both",
):

    panels = []
    labels = []

    if original_path and os.path.exists(original_path):
        panels.append(Image.open(original_path).convert("RGB"))
        labels.append("Original")

    if mode in ["relative", "both"] and severity_path and os.path.exists(severity_path):

        sev_png, _ = make_severity_png(
            severity_path,
            original_path
        )

        panels.append(
            Image.open(sev_png).convert("RGB")
        )

        labels.append("Relative Severity")

    if mode in ["pseudo", "both"] and pseudo_bi_path and os.path.exists(pseudo_bi_path):

        bi_png, _ = make_pseudo_bi_png(
            pseudo_bi_path,
            original_path
        )

        panels.append(
            Image.open(bi_png).convert("RGB")
        )

        labels.append("Pseudo-BI")

    if not panels:
        return None

    target_h = min(
        min(p.height for p in panels),
        950
    )

    def resize_keep_ratio(img, h):

        scale = h / img.height

        return img.resize(
            (
                int(img.width * scale),
                h
            ),
            Image.LANCZOS
        )

    panels = [
        resize_keep_ratio(p, target_h)
        for p in panels
    ]

    margin = 50
    gap = 40

    # HUGE TITLE SPACE
    title_h = 360
    label_h = 90

    out_w = (
        sum(p.width for p in panels)
        + gap * (len(panels) - 1)
        + margin * 2
    )

    out_h = (
        target_h
        + title_h
        + label_h
        + margin * 2
    )

    canvas = Image.new(
        "RGB",
        (out_w, out_h),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    # HUGE FONTS
    font_title = safe_font(150, bold=True)
    font_sub = safe_font(80, bold=True)
    font_label = safe_font(58, bold=True)

    title = f"{location} | {date}"

    if display_name and display_name != date:

        if "|" in display_name:
            suffix = display_name.split("|")[-1].strip()
            title += f" | {suffix}"

    # CENTER TITLE
    bbox = draw.textbbox(
        (0, 0),
        title,
        font=font_title
    )

    title_w = bbox[2] - bbox[0]

    title_x = (out_w - title_w) // 2

    draw.text(
        (title_x, 35),
        title,
        fill="black",
        font=font_title
    )

    subtitle = "HAB Bloom Analysis"

    bbox2 = draw.textbbox(
        (0, 0),
        subtitle,
        font=font_sub
    )

    sub_w = bbox2[2] - bbox2[0]

    sub_x = (out_w - sub_w) // 2

    draw.text(
        (sub_x, 190),
        subtitle,
        fill=(70, 70, 70),
        font=font_sub
    )

    x = margin

    y_label = margin + title_h

    y_img = y_label + label_h

    for panel, label in zip(panels, labels):

        bbox3 = draw.textbbox(
            (0, 0),
            label,
            font=font_label
        )

        label_w = bbox3[2] - bbox3[0]

        label_x = x + (
            panel.width - label_w
        ) // 2

        draw.text(
            (label_x, y_label),
            label,
            fill="black",
            font=font_label
        )

        canvas.paste(
            panel,
            (x, y_img)
        )

        x += panel.width + gap

    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    )

    canvas.save(tmp.name)

    return tmp.name


def make_mp4_from_items(items, location, mode, fps=1):
    frame_paths = []

    for item in items:
        if item["original_path"] is None:
            continue

        if mode == "original_relative" and item["severity_path"] is None:
            continue

        if mode == "original_pseudo" and item["pseudo_bi_path"] is None:
            continue

        png_path = create_side_by_side_download(
            original_path=item["original_path"],
            severity_path=item["severity_path"],
            pseudo_bi_path=item["pseudo_bi_path"],
            date=item["date"],
            location=location,
            display_name=item["display_name"],
            mode="relative" if mode == "original_relative" else "pseudo",
        )

        if png_path:
            frame_paths.append(png_path)

    if not frame_paths:
        return None

    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    max_w = max(im.width for im in frames)
    max_h = max(im.height for im in frames)

    # ffmpeg prefers even dimensions
    max_w += max_w % 2
    max_h += max_h % 2

    norm_frames = []
    for im in frames:
        canvas = Image.new("RGB", (max_w, max_h), "white")
        canvas.paste(im, ((max_w - im.width) // 2, (max_h - im.height) // 2))
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        canvas.save(out.name)
        norm_frames.append(out.name)

    tmp_dir = tempfile.mkdtemp()
    list_path = os.path.join(tmp_dir, "frames.txt")

    with open(list_path, "w", encoding="utf-8") as f:
        for p in norm_frames:
            f.write(f"file '{p.replace(chr(92), '/')}'\n")
            f.write(f"duration {1 / fps:.3f}\n")
        f.write(f"file '{norm_frames[-1].replace(chr(92), '/')}'\n")

    mp4_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-vsync",
        "vfr",
        "-pix_fmt",
        "yuv420p",
        mp4_path,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return mp4_path
    except Exception:
        # fallback gif-compatible mp4 creation with imageio
        try:
            import imageio.v2 as imageio
            writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8)
            for p in norm_frames:
                writer.append_data(np.array(Image.open(p).convert("RGB")))
            writer.close()
            return mp4_path
        except Exception as e:
            st.error(f"Could not create MP4. Install ffmpeg or imageio[ffmpeg]. Error: {e}")
            return None


# ============================================================
# MAP HELPERS
# ============================================================
def make_legend_html():
    sev_grad = "linear-gradient(to top,rgb(0,80,220),rgb(255,235,0),rgb(220,0,0))"
    bi_grad = (
        "linear-gradient(to top,"
        "rgb(0,80,60),rgb(0,130,80),rgb(80,185,50),"
        "rgb(220,220,0),rgb(255,140,0),rgb(200,30,0),rgb(100,0,0))"
    )
    return f"""
    <div style="position:fixed;bottom:30px;right:30px;z-index:9999;
    background:white;padding:12px;border:2px solid #444;border-radius:8px;
    font-size:13px;width:210px;box-shadow:0 2px 8px rgba(0,0,0,0.25);">
      <b>Relative Severity</b>
      <div style="display:flex;gap:10px;margin:6px 0 12px 0;">
        <div style="width:16px;height:90px;background:{sev_grad};border:1px solid #333;"></div>
        <div style="font-size:11px;display:flex;flex-direction:column;justify-content:space-between;height:90px;">
          <div><b>High</b></div><div><b>Medium</b></div><div><b>Low</b></div>
        </div>
      </div>
      <b>Pseudo Bloom Index</b> <span style="font-size:11px;color:#555;">({BI_MIN}–{BI_MAX})</span>
      <div style="display:flex;gap:10px;margin:6px 0 6px 0;">
        <div style="width:16px;height:140px;background:{bi_grad};border:1px solid #333;"></div>
        <div style="font-size:11px;display:flex;flex-direction:column;justify-content:space-between;height:140px;">
          <div><b>Extreme</b></div><div><b>Very High</b></div><div><b>High</b></div>
          <div><b>Medium</b></div><div><b>Low</b></div><div><b>Clean</b></div>
        </div>
      </div>
    </div>
    """


def add_layer_control_scroll(m, max_height="450px"):
    css = f"""
    <style>
    .leaflet-control-layers-expanded {{
        max-height:{max_height}!important;
        overflow-y:auto!important;
        overflow-x:hidden!important;
        padding-right:8px!important;
    }}
    .leaflet-control-layers-list {{
        max-height:{max_height}!important;
        overflow-y:auto!important;
    }}
    .leaflet-control-layers-overlays label,
    .leaflet-control-layers-base label {{
        white-space:nowrap;
        font-size:13px;
    }}
    </style>
    """
    m.get_root().header.add_child(folium.Element(css))


def make_timeline_chart(summary_df):
    chart_df = summary_df.copy().sort_values(["date_dt", "display_name"])

    rows = []
    for _, r in chart_df.iterrows():
        if not np.isnan(r.get("mean_pseudo_bi", float("nan"))):
            rows.append({
                "date_label": r["display_name"],
                "date_sort": r["sort_key"],
                "value": r["mean_pseudo_bi"],
            })

    if not rows:
        st.info("No Pseudo-BI data available for the timeline.")
        return

    spec = {
        "data": {"values": rows},
        "width": 360,
        "height": 240,
        "layer": [
            {
                "mark": {"type": "line", "opacity": 0.7, "color": "#00a86b"},
                "encoding": {
                    "x": {
                        "field": "date_label",
                        "type": "ordinal",
                        "sort": {"field": "date_sort", "order": "ascending"},
                        "axis": {"title": "Date"},
                    },
                    "y": {
                        "field": "value",
                        "type": "quantitative",
                        "axis": {"title": f"Pseudo-BI ({BI_MIN}–{BI_MAX})"},
                        "scale": {"domain": [BI_MIN, BI_MAX]},
                    },
                },
            },
            {
                "mark": {"type": "circle", "size": 140, "color": "#00a86b"},
                "encoding": {
                    "x": {
                        "field": "date_label",
                        "type": "ordinal",
                        "sort": {"field": "date_sort", "order": "ascending"},
                    },
                    "y": {"field": "value", "type": "quantitative"},
                    "tooltip": [
                        {"field": "date_label", "type": "ordinal", "title": "Date"},
                        {"field": "value", "type": "quantitative", "title": "Pseudo-BI", "format": ".3f"},
                    ],
                },
            },
        ],
    }

    st.vega_lite_chart(spec, use_container_width=True)


# ============================================================
# FILE DISCOVERY + MATCHING
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
        if f.startswith(f"{selected_body}/{prefix}/") and f.lower().endswith(".jpg")
    ])


original_files = get_jpgs("original")
severity_files = get_jpgs("severity")
pseudo_bi_files = get_jpgs("pseudo_bi")

if not severity_files:
    st.warning("No severity heatmaps found for this water body.")
    st.stop()


def product_preference_score(name):
    """Prefer final product files over intermediate same-date/same-image variants."""
    n = name.lower()
    score = 0
    if "_severity" in n:
        score += 10
    if "pseudo" in n or "pseudobi" in n or "pseudo_bi" in n:
        score += 10
    if "final" in n:
        score += 2
    return score


def download_with_sidecars(hf_paths):
    """
    Download files and de-duplicate technical duplicates.

    Important:
    Some folders contain two files for the same image/date, for example:
    ...FINAL_severity_index_georef.jpg
    ...FINAL_severity_index_georef_severity.jpg

    They should NOT become A/B duplicates. A/B is used only when there are
    two genuinely different images on the same date, for example DJI_0787 and DJI_0849.
    """
    local_by_key = {}

    for hf_path in hf_paths:
        date = extract_date_label(hf_path)
        name = os.path.splitext(os.path.basename(hf_path))[0]
        norm = normalize_name_for_matching(name)
        key = f"{date}__{norm}"

        local_jpg = download_file(HF_REPO_ID, hf_path, HF_REPO_TYPE, HF_TOKEN)

        for ext in [".jgw", ".prj"]:
            sidecar = os.path.splitext(hf_path)[0] + ext
            if sidecar in all_files:
                download_file(HF_REPO_ID, sidecar, HF_REPO_TYPE, HF_TOKEN)

        item = {
            "key": key,
            "path": local_jpg,
            "date": date,
            "name": name,
            "norm": norm,
            "hf_path": hf_path,
        }

        # Keep only one file per normalized product key.
        # If duplicate technical variants exist, keep the more final/product-like one.
        if key not in local_by_key:
            local_by_key[key] = item
        else:
            old_score = product_preference_score(local_by_key[key]["name"])
            new_score = product_preference_score(name)
            if new_score >= old_score:
                local_by_key[key] = item

    # Build date index AFTER de-duplication.
    local_by_date = {}
    for item in local_by_key.values():
        local_by_date.setdefault(item["date"], []).append(item)

    for date in local_by_date:
        local_by_date[date] = sorted(local_by_date[date], key=lambda x: x["name"])

    return local_by_key, local_by_date


def find_matching_item(source_item, by_key, by_date):
    # exact normalized key match first
    if source_item["key"] in by_key:
        return by_key[source_item["key"]]

    candidates = by_date.get(source_item["date"], [])
    if not candidates:
        return None

    # best fuzzy match by longest shared normalized prefix
    src_norm = source_item["norm"]
    best = None
    best_score = -1

    for cand in candidates:
        c_norm = cand["norm"]

        # common prefix score
        score = 0
        for a, b in zip(src_norm, c_norm):
            if a == b:
                score += 1
            else:
                break

        # fallback if DJI/date token is shared
        dji = re.search(r"dji_\d+", src_norm)
        if dji and dji.group(0) in c_norm:
            score += 1000

        if score > best_score:
            best_score = score
            best = cand

    return best


with st.spinner("Downloading data..."):
    orig_by_key, orig_by_date = download_with_sidecars(original_files)
    sev_by_key, sev_by_date = download_with_sidecars(severity_files)
    pseudo_bi_by_key, pseudo_bi_by_date = download_with_sidecars(pseudo_bi_files)


# ============================================================
# LABELS: ONLY ADD A/B WHEN DATE HAS DUPLICATES
# ============================================================
date_counts = {date: len(items) for date, items in sev_by_date.items()}
date_seen = {}

all_items = []
for date, items in sev_by_date.items():
    for item in items:
        date_seen.setdefault(date, 0)
        idx = date_seen[date]
        date_seen[date] += 1

        if date_counts[date] > 1:
            label = f"{date} | {chr(ord('A') + idx)}"
        else:
            label = date

        item = dict(item)
        item["display_name"] = label
        item["sort_key"] = f"{parse_date(date).strftime('%Y-%m-%d') if pd.notna(parse_date(date)) else date}_{idx:03d}"
        item["dup_index"] = idx
        all_items.append(item)

all_items = sorted(all_items, key=lambda x: x["sort_key"])

if not all_items:
    st.error("No dated files found.")
    st.stop()


# ============================================================
# BUILD MAP
# ============================================================
first_bounds = read_jgw_bounds(all_items[0]["path"])
center_lat = (first_bounds[0][0] + first_bounds[1][0]) / 2
center_lon = (first_bounds[0][1] + first_bounds[1][1]) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles=None)

folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Esri World Imagery",
    control=True,
).add_to(m)

summary_rows = []
matched_items_for_download = []

for i, sev_item in enumerate(all_items):
    show_first = (i == 0)

    date = sev_item["date"]
    display_name = sev_item["display_name"]

    bounds = read_jgw_bounds(sev_item["path"])

    orig_item = find_matching_item(sev_item, orig_by_key, orig_by_date)
    bi_item = find_matching_item(sev_item, pseudo_bi_by_key, pseudo_bi_by_date)

    orig_path = orig_item["path"] if orig_item else None
    sev_path = sev_item["path"]
    bi_path = bi_item["path"] if bi_item else None

    if orig_path:
        ImageOverlay(
            name=f"{display_name} | Original",
            image=make_original_png(orig_path),
            bounds=bounds,
            opacity=1.0,
            interactive=True,
            show=show_first,
        ).add_to(m)

    sev_png, mean_sev = make_severity_png(sev_path, orig_path)
    ImageOverlay(
        name=f"{display_name} | Severity",
        image=sev_png,
        bounds=bounds,
        opacity=1.0,
        interactive=True,
        show=False,
    ).add_to(m)

    mean_bi = float("nan")
    if bi_path:
        bi_png, mean_bi = make_pseudo_bi_png(bi_path, orig_path)
        ImageOverlay(
            name=f"{display_name} | Pseudo-BI",
            image=bi_png,
            bounds=bounds,
            opacity=1.0,
            interactive=True,
            show=False,
        ).add_to(m)

    summary_rows.append({
        "key": sev_item["key"],
        "date": date,
        "display_name": display_name,
        "sort_key": sev_item["sort_key"],
        "date_dt": parse_date(date),
        "mean_severity": mean_sev,
        "severity_level": severity_category(mean_sev),
        "mean_pseudo_bi": mean_bi,
        "bi_level": bi_category(mean_bi) if not np.isnan(mean_bi) else "—",
    })

    matched_items_for_download.append({
        "key": sev_item["key"],
        "date": date,
        "display_name": display_name,
        "sort_key": sev_item["sort_key"],
        "original_path": orig_path,
        "severity_path": sev_path,
        "pseudo_bi_path": bi_path,
    })

m.get_root().html.add_child(folium.Element(make_legend_html()))
add_layer_control_scroll(m, max_height="450px")
folium.LayerControl(collapsed=False).add_to(m)


# ============================================================
# DISPLAY APP
# ============================================================
col1, col2 = st.columns([3, 1])

with col1:
    st.caption("Use the layer control to toggle dates and products.")
    st_folium(m, width=1000, height=700)

with col2:
    st.subheader("Timeline")

    summary_df = pd.DataFrame(summary_rows).sort_values("sort_key").reset_index(drop=True)

    display_cols = [
        "display_name",
        "mean_severity",
        "severity_level",
        "mean_pseudo_bi",
        "bi_level",
    ]

    st.dataframe(
        summary_df[display_cols].rename(columns={
            "display_name": "Date",
            "mean_severity": "Severity",
            "severity_level": "Sev. Level",
            "mean_pseudo_bi": "Pseudo-BI",
            "bi_level": "BI Level",
        }),
        hide_index=True,
        use_container_width=True,
    )

    make_timeline_chart(summary_df)

    latest = summary_df.iloc[-1]
    st.metric("Latest Date", latest["display_name"])
    st.metric("Latest Severity Level", latest["severity_level"])
    st.metric("Latest Mean Severity", f"{latest['mean_severity']:.2f}")

    if not np.isnan(latest["mean_pseudo_bi"]):
        st.metric("Latest BI Level", latest["bi_level"])
        st.metric("Latest Pseudo-BI", f"{latest['mean_pseudo_bi']:.2f}")

    st.divider()
    st.subheader("Download Image")

    downloadable_items = [x for x in matched_items_for_download if x["original_path"] and x["severity_path"]]

    if downloadable_items:
        labels = [x["display_name"] for x in downloadable_items]

        selected_label = st.selectbox(
            "Choose date/image to download",
            labels,
            index=len(labels) - 1,
        )

        selected_item = downloadable_items[labels.index(selected_label)]

        image_mode = st.radio(
            "Image export content",
            ["Original + Relative", "Original + Pseudo-BI", "Original + Relative + Pseudo-BI"],
            index=2,
        )

        mode_map = {
            "Original + Relative": "relative",
            "Original + Pseudo-BI": "pseudo",
            "Original + Relative + Pseudo-BI": "both",
        }

        preview_path = create_side_by_side_download(
            original_path=selected_item["original_path"],
            severity_path=selected_item["severity_path"],
            pseudo_bi_path=selected_item["pseudo_bi_path"],
            date=selected_item["date"],
            location=selected_body,
            display_name=selected_item["display_name"],
            mode=mode_map[image_mode],
        )

        if preview_path:
            st.image(preview_path, caption=f"Preview ({selected_item['display_name']})", use_container_width=True)

            with open(preview_path, "rb") as f:
                st.download_button(
                    label="Download PNG",
                    data=f,
                    file_name=f"{selected_body}_{selected_item['display_name'].replace(' | ', '_')}_analysis.png",
                    mime="image/png",
                )
    else:
        st.info("No matching images found for download.")

    st.divider()
    st.subheader("Download MP4 Video")

    video_mode = st.radio(
        "Video export content",
        ["Original + Relative", "Original + Pseudo-BI"],
        index=0,
        key="video_mode_radio",
    )

    fps = st.slider("Frames per second", min_value=1, max_value=5, value=1)

    if st.button("Create MP4"):
        with st.spinner("Creating MP4 video..."):
            mode = "original_relative" if video_mode == "Original + Relative" else "original_pseudo"
            mp4_path = make_mp4_from_items(matched_items_for_download, selected_body, mode, fps=fps)

        if mp4_path:
            with open(mp4_path, "rb") as f:
                st.download_button(
                    label="Download MP4",
                    data=f,
                    file_name=f"{selected_body}_{video_mode.replace(' + ', '_').replace(' ', '_')}.mp4",
                    mime="video/mp4",
                )
