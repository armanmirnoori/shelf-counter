"""Core logic: detect products on a shelf photo, group them into shelf rows, and
estimate how many units sit behind each visible facing using a depth model."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.ops import nms
from ultralytics import YOLO, YOLOWorld

ROOT = Path(__file__).resolve().parent

# Text prompts for the open-vocabulary detector (YOLO-World only).
DEFAULT_PROMPTS = ["product", "bottle", "box", "can", "package", "jar", "carton", "bag"]

# The shelf detector is a YOLO11 trained on SKU-110K (densely packed retail shelves).
# It is far more accurate on shelves than open-vocabulary YOLO-World, but only knows
# one class ("object"); product shapes are then labelled with CLIP (see ProductTyper).
SKU_REPO = "chistopat/sku110k-yolo11-object-detector"
SKU_FILE = "weights/sku110k-yolo11-s640.onnx"
MODELS = {
    "Shelf detector (SKU-110K)": "sku110k",
    "Open vocabulary (YOLO-World S)": "yolov8s-worldv2.pt",
    "Open vocabulary (YOLO-World L)": "yolov8l-worldv2.pt",
}
DEFAULT_MODEL = "Shelf detector (SKU-110K)"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
PRODUCT_TYPES = ["bottle", "jar", "can", "box", "bag"]

ROW_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
              "#42d4f4", "#f032e6", "#bfef45", "#469990", "#9a6324"]

ROUND_WORDS = ("bottle", "can", "jar", "tin", "cup", "tube", "roll")
FLAT_DEPTH_TO_HEIGHT = 0.25   # boxes and bags are roughly a quarter as deep as tall


def is_round(name: str) -> bool:
    return any(w in name.lower() for w in ROUND_WORDS)


@dataclass
class CountResult:
    boxes: np.ndarray            # (N, 4) boxes as x1, y1, x2, y2
    scores: np.ndarray           # (N,) confidence per box
    labels: list = field(default_factory=list)   # prompt word per box
    rows: list = field(default_factory=list)     # shelf-row number (1 = top) per box
    units: Optional[np.ndarray] = None           # estimated units deep per facing

    @property
    def total(self) -> int:
        """Visible facings."""
        return len(self.boxes)

    @property
    def estimated_total(self) -> Optional[int]:
        """Facings x estimated units behind each one (None if depth was not run)."""
        return None if self.units is None else int(self.units.sum())

    def row_counts(self) -> dict:
        counts = {}
        for r in self.rows:
            counts[r] = counts.get(r, 0) + 1
        return dict(sorted(counts.items()))

    def row_units(self) -> dict:
        totals = {}
        if self.units is not None:
            for r, u in zip(self.rows, self.units):
                totals[r] = totals.get(r, 0) + int(u)
        return dict(sorted(totals.items()))


# --------------------------------------------------------------------------- rows

def assign_rows(boxes: np.ndarray) -> list:
    """Group boxes into shelf rows.

    Products on one shelf stand on the same surface, so neighbours share a bottom
    edge. Each box is linked to its nearest left-hand neighbours whose bottom edge
    lines up; chains of links form a row. Because only *neighbours* are compared,
    this follows shelves that slant across the photo (perspective shots).
    """
    n = len(boxes)
    if n == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    med_w = float(np.median(w))

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        min_h = np.minimum(h[i], h)
        overlap = np.minimum(y2[i], y2) - np.maximum(y1[i], y1)
        gap = x1[i] - x2                       # horizontal gap to boxes on the left
        ok = ((cx < cx[i]) & (gap < 2.0 * med_w)
              & (np.abs(y2 - y2[i]) < 0.35 * min_h) & (overlap > 0.5 * min_h))
        ok[i] = False
        cand = np.where(ok)[0]
        for j in cand[np.argsort(-cx[cand])][:2]:   # two nearest neighbours
            parent[find(i)] = find(int(j))
        # Items stacked directly on top of box i (e.g. two layers of cans) share its shelf.
        x_overlap = np.minimum(x2[i], x2) - np.maximum(x1[i], x1)
        stacked = ((x_overlap > 0.5 * np.minimum(w[i], w))
                   & (np.abs(y1[i] - y2) < 0.15 * min_h))
        stacked[i] = False
        for j in np.where(stacked)[0]:
            parent[find(i)] = find(int(j))

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    # Fold stray 1-2 box groups into the closest real row they overlap vertically.
    big = [g for g in groups.values() if len(g) >= 3]
    small = [g for g in groups.values() if len(g) < 3]
    if big:
        owner = {i: k for k, g in enumerate(big) for i in g}
        big_idx = np.array(sorted(owner))
        for g in small:
            for i in g:
                overlap = np.minimum(y2[i], y2[big_idx]) - np.maximum(y1[i], y1[big_idx])
                dist = np.hypot(cx[big_idx] - cx[i], cy[big_idx] - cy[i])
                dist[overlap < 0.3 * np.minimum(h[i], h[big_idx])] = np.inf
                j = int(np.argmin(dist))
                if np.isfinite(dist[j]) and dist[j] < 4 * med_w:
                    big[owner[int(big_idx[j])]].append(i)
                else:
                    big.append([i])
        final = big
    else:
        final = list(groups.values())

    final.sort(key=lambda g: float(np.median(cy[g])))
    rows = [0] * n
    for r, g in enumerate(final, start=1):
        for i in g:
            rows[i] = r
    return rows


# ---------------------------------------------------------------------- detection

def _tile_grid(w: int, h: int, n: int, overlap: float = 0.25):
    """n x n overlapping tiles as (x1, y1, x2, y2)."""
    tw, th = w / (n - (n - 1) * overlap), h / (n - (n - 1) * overlap)
    tiles = []
    for r in range(n):
        for c in range(n):
            tx, ty = c * tw * (1 - overlap), r * th * (1 - overlap)
            tiles.append((int(round(tx)), int(round(ty)),
                          min(w, int(round(tx + tw))), min(h, int(round(ty + th)))))
    return tiles


def _clean(boxes, scores, cls, iou=0.5):
    """NMS, then remove boxes that wrap several products and nested duplicates."""
    if len(boxes) == 0:
        return boxes, scores, cls
    keep = nms(torch.from_numpy(boxes).float(), torch.from_numpy(scores).float(), iou).numpy()
    boxes, scores, cls = boxes[keep], scores[keep], cls[keep]

    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    inside = ((cx[None] > boxes[:, 0:1]) & (cx[None] < boxes[:, 2:3])
              & (cy[None] > boxes[:, 1:2]) & (cy[None] < boxes[:, 3:4]))
    ratio = area[None] / area[:, None]            # area_j / area_i
    wraps = (inside & (ratio > 0.15) & (ratio < 0.6)).sum(1)
    keep = wraps < 2                              # box i covers 2+ smaller products
    boxes, scores, cls, area = boxes[keep], scores[keep], cls[keep], area[keep]

    # Greedy: a box mostly inside an already-kept, higher-scoring box is a duplicate.
    order = np.argsort(-scores)
    kept = []
    for i in order:
        if kept:
            k = np.array(kept)
            iw = np.minimum(boxes[i, 2], boxes[k, 2]) - np.maximum(boxes[i, 0], boxes[k, 0])
            ih = np.minimum(boxes[i, 3], boxes[k, 3]) - np.maximum(boxes[i, 1], boxes[k, 1])
            inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
            if (inter / np.minimum(area[i], area[k])).max() > 0.7:
                continue
        kept.append(int(i))
    kept = np.array(kept, dtype=int)
    return boxes[kept], scores[kept], cls[kept]


class ShelfCounter:
    def __init__(self, weights: str = MODELS[DEFAULT_MODEL], prompts=None):
        self.open_vocab = weights != "sku110k"
        if self.open_vocab:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = YOLOWorld(weights)  # weights download automatically on first run
            self.set_prompts(prompts or DEFAULT_PROMPTS)
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(SKU_REPO, SKU_FILE, local_dir=ROOT / "weights" / "sku110k")
            self.device = "cpu"                     # ONNX export; ~0.3 s per pass on CPU
            self.model = YOLO(path, task="detect")
            self.prompts = ["product"]

    def set_prompts(self, prompts):
        if self.open_vocab:
            self.prompts = list(prompts)
            self.model.set_classes(self.prompts)

    def _predict(self, image, conf, imgsz):
        if not self.open_vocab:
            imgsz = 640                             # the ONNX export has a fixed input size
        res = self.model.predict(image, conf=conf, imgsz=imgsz, iou=0.5, agnostic_nms=True,
                                 max_det=1000, verbose=False, device=self.device)[0]
        return (res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int))

    def count(self, image: Image.Image, conf: float = 0.15, imgsz: int = 960,
              tiles: int = 2, max_box_area: float = 0.25) -> CountResult:
        """Detect products. `tiles` > 1 also runs the detector on an n x n grid of
        overlapping crops, which finds the small, densely packed items a single
        pass over the whole photo misses."""
        image = image.convert("RGB")
        w, h = image.size
        all_b, all_s, all_c = [list(x) for x in zip(self._predict(image, conf, imgsz))]

        if tiles > 1:
            for tx1, ty1, tx2, ty2 in _tile_grid(w, h, tiles):
                b, s, c = self._predict(image.crop((tx1, ty1, tx2, ty2)), conf, imgsz)
                if len(b) == 0:
                    continue
                # Drop boxes cut off by a tile edge; a neighbouring tile sees them whole.
                m = 3
                cut = (((b[:, 0] <= m) & (tx1 > 0)) | ((b[:, 1] <= m) & (ty1 > 0))
                       | ((b[:, 2] >= tx2 - tx1 - m) & (tx2 < w))
                       | ((b[:, 3] >= ty2 - ty1 - m) & (ty2 < h)))
                b = b[~cut] + np.array([tx1, ty1, tx1, ty1], dtype=b.dtype)
                all_b.append(b), all_s.append(s[~cut]), all_c.append(c[~cut])

        boxes, scores, cls = np.concatenate(all_b), np.concatenate(all_s), np.concatenate(all_c)

        # Drop huge boxes: they are usually the whole shelf unit, not a product.
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = areas < max_box_area * w * h
        boxes, scores, cls = _clean(boxes[keep], scores[keep], cls[keep])

        return CountResult(boxes=boxes, scores=scores,
                           labels=[self.prompts[c] for c in cls], rows=assign_rows(boxes))


# ------------------------------------------------------------------ product types

class ProductTyper:
    """Labels each detected product as bottle / jar / can / box / bag with CLIP, so the
    depth estimate knows its shape (a can is as deep as it is wide, a box is not)."""

    def __init__(self):
        import clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device,
                                                download_root=str(ROOT / "weights" / "clip"))
        tokens = clip.tokenize([f"a photo of a {t} of food on a store shelf"
                                for t in PRODUCT_TYPES]).to(self.device)
        with torch.no_grad():
            text = self.model.encode_text(tokens).float()
        self.text = text / text.norm(dim=-1, keepdim=True)

    def __call__(self, image: Image.Image, boxes: np.ndarray, batch: int = 64) -> list:
        image = image.convert("RGB")
        labels = []
        for start in range(0, len(boxes), batch):
            crops = torch.stack([self.preprocess(image.crop(tuple(int(v) for v in b)))
                                 for b in boxes[start:start + batch]]).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(crops).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            labels += [PRODUCT_TYPES[i] for i in (feats @ self.text.T).argmax(1).tolist()]
        return labels


def smooth_labels(result: CountResult) -> list:
    """Majority vote among similar-sized neighbours on the same shelf: identical
    products stand side by side, so a lone odd label is almost always a CLIP mistake."""
    b = result.boxes
    if len(b) == 0:
        return []
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    cx = (b[:, 0] + b[:, 2]) / 2
    rows = np.array(result.rows)
    out = []
    for i in range(len(b)):
        near = ((rows == rows[i]) & (np.abs(h - h[i]) < 0.25 * h[i])
                & (np.abs(cx - cx[i]) < 3.5 * w[i]))
        votes = [result.labels[j] for j in np.where(near)[0]]
        out.append(max(set(votes), key=lambda v: (votes.count(v), v == result.labels[i])))
    return out


# -------------------------------------------------------------------------- depth

class DepthEstimator:
    """Monocular metric depth (Depth Anything V2): metres from the camera per pixel."""

    def __init__(self, model_id: str = DEPTH_MODEL):
        from transformers import pipeline
        self.pipe = pipeline("depth-estimation", model=model_id,
                             device=0 if torch.cuda.is_available() else -1)

    def __call__(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        depth = self.pipe(image)["predicted_depth"].squeeze().float()
        if tuple(depth.shape) != (image.height, image.width):
            depth = torch.nn.functional.interpolate(
                depth[None, None], size=(image.height, image.width),
                mode="bilinear", align_corners=False)[0, 0]
        return depth.cpu().numpy()


def focal_length_px(image: Image.Image, default_fov_deg: float = 65.0) -> float:
    """Focal length in pixels, from EXIF if the photo has it, else a typical phone FOV."""
    long_side = max(image.size)
    try:
        f35 = image.getexif().get_ifd(0x8769).get(0xA405)  # FocalLengthIn35mmFilm
        if f35:
            return float(f35) / 36.0 * long_side
    except Exception:
        pass
    return 0.5 * long_side / np.tan(np.radians(default_fov_deg) / 2)


def estimate_units(result: CountResult, depth: np.ndarray, focal_px: float,
                   shelf_depth_cm: float = 40.0, tallest_cm: float = 30.0,
                   max_units: int = 40) -> np.ndarray:
    """Estimate how many units stand behind each visible facing.

    1. Distance to each product = median depth inside its box.
    2. Size = pixel size x distance. The depth model gets relative sizes right but
       its absolute scale drifts, so sizes are calibrated so that the tallest
       products in the photo are `tallest_cm` high.
    3. Round products (cans, jars, bottles) are as deep as they are wide; boxes and
       bags are about a quarter as deep as they are tall (height is used because
       width shrinks in angled shots).
    4. Products standing further back than their row's front line leave less room
       behind them, so that recess is subtracted from the shelf depth.
    5. Units deep = usable shelf depth / product depth.
    """
    n = len(result.boxes)
    if n == 0:
        return np.zeros(0, dtype=int)
    H, W = depth.shape
    z = np.zeros(n)
    for i, (x1, y1, x2, y2) in enumerate(result.boxes):
        qx, qy = (x2 - x1) / 4, (y2 - y1) / 4           # central half of the box
        patch = depth[int(max(0, y1 + qy)):int(min(H, y2 - qy)) + 1,
                      int(max(0, x1 + qx)):int(min(W, x2 - qx)) + 1]
        z[i] = np.median(patch) if patch.size else np.nan
    z = np.where(np.isfinite(z), z, np.nanmedian(z))

    raw_h = (result.boxes[:, 3] - result.boxes[:, 1]) * z
    scale = (tallest_cm / 100.0) / np.percentile(raw_h, 95)      # metres per (px x depth)
    width_m = (result.boxes[:, 2] - result.boxes[:, 0]) * z * scale
    height_m = raw_h * scale
    z = z * scale * focal_px                                     # calibrated distance
    round_ = np.array([is_round(name) for name in result.labels])
    product_depth = np.maximum(0.04,np.where(round_, width_m, FLAT_DEPTH_TO_HEIGHT * height_m))

    # Recess behind the row's front line. A shelf edge receding in perspective is a
    # straight line in inverse depth (1/z) against image x, so the fit is done there.
    cx = (result.boxes[:, 0] + result.boxes[:, 2]) / 2
    rows = np.array(result.rows)
    recess = np.zeros(n)
    for r in np.unique(rows):
        idx = np.where(rows == r)[0]
        if len(idx) < 4:
            continue
        use = np.ones(len(idx), dtype=bool)
        for _ in range(3):                               # refit on the front-most items
            coef = np.polyfit(cx[idx][use], 1.0 / z[idx][use], 1)
            front = 1.0 / np.clip(np.polyval(coef, cx[idx]), 1e-3, None)
            resid = z[idx] - front
            use = resid <= np.median(resid)
            if use.sum() < 2:
                break
        recess[idx] = np.clip(resid - 0.05, 0, None)     # ignore < 5 cm (depth noise)

    usable = np.clip(shelf_depth_cm / 100.0 - recess, 0, None)
    return np.clip(np.floor(usable / product_depth), 1, max_units).astype(int)


# ------------------------------------------------------------------------ drawing

def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                    # older Pillow
        return ImageFont.load_default()


def draw(image: Image.Image, result: CountResult, show_rows: bool = True) -> Image.Image:
    """Return a copy of the image with boxes drawn (coloured by row if requested).
    When depth units are available each box is labelled with its units-deep estimate."""
    out = image.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    lw = max(2, out.width // 500)
    fs = max(12, out.width // 90)
    font = _font(fs)
    for i, (box, row) in enumerate(zip(result.boxes, result.rows)):
        color = ROW_COLORS[(row - 1) % len(ROW_COLORS)] if show_rows else "#00ff00"
        d.rectangle(box.tolist(), outline=color, width=lw)
        if result.units is not None:
            text = f"x{int(result.units[i])}"
            tw = d.textlength(text, font=font)
            if tw + 6 < box[2] - box[0]:                 # skip labels on tiny boxes
                d.rectangle([box[0], box[1], box[0] + tw + 6, box[1] + fs + 4], fill=color)
                d.text((box[0] + 3, box[1] + 1), text, fill="white", font=font)
    if show_rows:
        for row in result.row_counts():
            idx = [i for i, r in enumerate(result.rows) if r == row]
            left = min(idx, key=lambda i: result.boxes[i][0])
            x, y = result.boxes[left][0], result.boxes[left][3]
            text = f"Row {row}"
            tw = d.textlength(text, font=font)
            d.rectangle([x, y - fs - 6, x + tw + 8, y], fill=ROW_COLORS[(row - 1) % len(ROW_COLORS)])
            d.text((x + 4, y - fs - 4), text, fill="white", font=font)
    return out


def depth_preview(depth: np.ndarray) -> Image.Image:
    """Depth map as a colour image (near = bright yellow, far = dark blue)."""
    inv = 1.0 / np.clip(depth, 1e-3, None)               # inverse depth keeps near detail
    lo, hi = np.percentile(inv, [2, 98])
    norm = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    grey = ImageOps.equalize(Image.fromarray((norm * 255).astype(np.uint8)))
    return ImageOps.colorize(grey, black="#0d0887", mid="#cc4778", white="#f0f921")
