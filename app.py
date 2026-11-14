"""Streamlit dashboard: upload shelf photos, get visible facings, per-row counts and a
depth-based estimate of the total stock (including units hidden behind the front row)."""
import dataclasses
import io

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from counter import (DEFAULT_MODEL, DEFAULT_PROMPTS, MODELS, DepthEstimator, ProductTyper,
                     ShelfCounter, depth_preview, draw, estimate_units, focal_length_px,
                     smooth_labels)

st.set_page_config(page_title="Shelf Item Counter", page_icon="🛒", layout="wide")


@st.cache_resource(show_spinner="Loading detection model...")
def load_counter(weights: str):
    return ShelfCounter(weights)


@st.cache_resource(show_spinner="Loading depth model...")
def load_depth():
    return DepthEstimator()


@st.cache_resource(show_spinner="Loading product-shape model...")
def load_typer():
    return ProductTyper()


def open_image(data: bytes) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data)))  # respect phone rotation


@st.cache_data(show_spinner=False, max_entries=20)
def detect(data: bytes, weights: str, prompts: tuple, conf: float, imgsz: int, tiles: int):
    counter = load_counter(weights)
    if list(prompts) != counter.prompts:
        counter.set_prompts(prompts)
    return counter.count(open_image(data), conf=conf, imgsz=imgsz, tiles=tiles)


@st.cache_data(show_spinner=False, max_entries=20)
def product_types(data: bytes, weights: str, prompts: tuple, conf: float, imgsz: int, tiles: int):
    result = detect(data, weights, prompts, conf, imgsz, tiles)
    labels = load_typer()(open_image(data), result.boxes)
    return smooth_labels(dataclasses.replace(result, labels=labels))


@st.cache_data(show_spinner=False, max_entries=10)
def depth_map(data: bytes):
    return load_depth()(open_image(data))


st.title("🛒 Shelf Item Counter")
st.caption("Upload photos of retail shelves. The model detects every visible product, groups "
           "them by shelf row, and uses a depth model to estimate the units stocked behind "
           "each one.")

with st.sidebar:
    st.header("Detection")
    model_name = st.selectbox("Model", list(MODELS), index=list(MODELS).index(DEFAULT_MODEL),
                              help="Larger models find more products but are slower. "
                                   "Each one downloads once on first use.")
    open_vocab = MODELS[model_name] != "sku110k"
    conf = st.slider("Detection confidence", 0.05, 0.6, 0.15 if open_vocab else 0.25, 0.01,
                     help="Lower = finds more items (may add false positives).")
    tiles = st.select_slider("Tiling", [1, 2, 3], value=2,
                             format_func=lambda n: "Off" if n == 1 else f"{n} x {n}",
                             help="Also scans the photo in overlapping crops. Finds small, "
                                  "densely packed products; slower.")
    imgsz = st.select_slider("Inference resolution", [640, 960, 1280, 1600], value=960,
                             disabled=not open_vocab,
                             help="YOLO-World only. Applied to the photo and to every tile.")
    prompts = st.text_input("What to look for (comma separated)", ", ".join(DEFAULT_PROMPTS),
                            disabled=not open_vocab, help="YOLO-World only.")
    show_rows = st.toggle("Group by shelf row", value=True)
    show_boxes = st.toggle("Show detection boxes", value=True)

    st.header("Depth")
    use_depth = st.toggle("Estimate units behind each item", value=True)
    shelf_depth = st.slider("Shelf depth (cm)", 15, 80, 40, 5, disabled=not use_depth,
                            help="Front-to-back depth of the shelf. Set this to the real "
                                 "value for your fixtures.")
    tallest = st.slider("Tallest product in photo (cm)", 10, 60, 30, 1, disabled=not use_depth,
                        help="Used to calibrate real-world sizes. A cereal box is about "
                             "30 cm, a soup can about 10 cm.")

wanted = tuple(p.strip() for p in prompts.split(",") if p.strip()) or tuple(DEFAULT_PROMPTS)

files = st.file_uploader("Shelf images", type=["jpg", "jpeg", "png", "webp"],
                         accept_multiple_files=True)
if not files:
    st.info("Upload one or more shelf photos to start.")
    st.stop()

summary = []
for f in files:
    data = f.getvalue()
    image = open_image(data)
    with st.spinner(f"Detecting items in {f.name}..."):
        result = detect(data, MODELS[model_name], wanted, conf, imgsz, tiles)
    depth = None
    if use_depth and result.total:
        with st.spinner(f"Estimating depth in {f.name}..."):
            depth = depth_map(data)
        if MODELS[model_name] == "sku110k":      # one-class detector: get shapes from CLIP
            with st.spinner(f"Identifying product shapes in {f.name}..."):
                result = dataclasses.replace(result, labels=product_types(
                    data, MODELS[model_name], wanted, conf, imgsz, tiles))
        result = dataclasses.replace(result, units=estimate_units(
            result, depth, focal_length_px(image), shelf_depth_cm=shelf_depth,
            tallest_cm=tallest))

    record = {"image": f.name, "visible_facings": result.total}
    if result.units is not None:
        record["estimated_total"] = result.estimated_total
    if show_rows:
        record.update({f"row_{r}": c for r, c in result.row_counts().items()})
    summary.append(record)

    st.divider()
    left, right = st.columns([3, 1])
    with left:
        shown = draw(image, result, show_rows) if show_boxes else image
        st.image(shown, caption=f.name, use_container_width=True)
        if depth is not None:
            with st.expander("Depth map (yellow = near, blue = far)"):
                st.image(depth_preview(depth), use_container_width=True)
    with right:
        st.metric("Visible facings", result.total)
        if result.units is not None:
            st.metric("Estimated total (with depth)", result.estimated_total,
                      help="Sum of the xN labels: each visible item times the units "
                           "estimated to fit behind it on a full shelf.")
        if show_rows and result.total:
            table = {"Row (top → bottom)": list(result.row_counts().keys()),
                     "Facings": list(result.row_counts().values())}
            if result.units is not None:
                table["Est. units"] = list(result.row_units().values())
            st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
        if result.units is not None:
            st.caption("Units behind the front row cannot be seen in a photo. The estimate "
                       "assumes shelves are stocked to the back, and is derived from the "
                       "shelf depth and each product's estimated size. Treat it as a "
                       "rough figure, not a stock count.")

if len(summary) > 1 or show_rows:
    st.divider()
    st.subheader("Summary")
    df = pd.DataFrame(summary).fillna(0)
    st.dataframe(df, hide_index=True, use_container_width=True)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Download CSV", buf.getvalue(), "shelf_counts.csv", "text/csv")
