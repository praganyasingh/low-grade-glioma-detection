"""Neurosymbolic LGG Segmentation & Analysis — web backend.
"""
import base64
import io
import os

import cv2
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image
from skimage.segmentation import mark_boundaries



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(BASE_DIR, "..", "model", "fpn_segmentation_weights.pth")
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIXEL_SPACING_MM = 0.5

ALLOWED_EXTENSIONS = {"tif", "tiff", "png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB uploads

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CustomBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ConvBlock(3, 64, 3, 2, 1)
        self.layer2 = ConvBlock(64, 128, 3, 2, 1)
        self.layer3 = ConvBlock(128, 256, 3, 2, 1)
        self.layer4 = ConvBlock(256, 512, 3, 2, 1)

    def forward(self, x):
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4


class FPN(nn.Module):
    def __init__(self, in_channels=(64, 128, 256, 512), out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(ch, out_channels, 1) for ch in in_channels])
        self.smooth_convs = nn.ModuleList([ConvBlock(out_channels, out_channels, 3, 1, 1) for _ in in_channels])

    def forward(self, features):
        p4 = self.lateral_convs[3](features[3])
        p3 = self._upsample_add(p4, self.lateral_convs[2](features[2]))
        p2 = self._upsample_add(p3, self.lateral_convs[1](features[1]))
        p1 = self._upsample_add(p2, self.lateral_convs[0](features[0]))
        return [self.smooth_convs[i](p) for i, p in enumerate([p1, p2, p3, p4])]

    @staticmethod
    def _upsample_add(x, y):
        return F.interpolate(x, size=y.shape[2:], mode="bilinear", align_corners=False) + y


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)


class FPN_Segmentation(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = CustomBackbone()
        self.fpn = FPN()
        self.head = SegmentationHead(256, 128, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        return self.head(fpn_features[0])  


print(f"Loading model from {MODEL_PATH} on {DEVICE} ...")
model = FPN_Segmentation(num_classes=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model ready.")

def build_knowledge_graph() -> nx.DiGraph:
    G = nx.DiGraph()

    for n in ["Small Tumor", "Medium Tumor", "Large Tumor"]:
        G.add_node(n, type="Tumor")

    for n in ["Cognitive Impairment", "Balance Issues", "Headaches"]:
        G.add_node(n, type="Symptom")

    for n in ["Surgery", "Radiotherapy", "Chemotherapy"]:
        G.add_node(n, type="Treatment")

    for n in ["High Survival Rate", "Long Recovery Time", "Low Recurrence Risk"]:
        G.add_node(n, type="Outcome")

    G.add_edge("Small Tumor", "Radiotherapy", relation="requires_treatment")
    G.add_edge("Medium Tumor", "Radiotherapy", relation="requires_treatment")
    G.add_edge("Large Tumor", "Surgery", relation="requires_treatment")

    G.add_edge("Cognitive Impairment", "Small Tumor", relation="associated_with")
    G.add_edge("Balance Issues", "Medium Tumor", relation="associated_with")
    G.add_edge("Headaches", "Large Tumor", relation="associated_with")

    G.add_edge("Surgery", "High Survival Rate", relation="follows")
    G.add_edge("Radiotherapy", "Long Recovery Time", relation="follows")
    G.add_edge("Chemotherapy", "Low Recurrence Risk", relation="follows")

    return G


KG = build_knowledge_graph()
_GRAPH_LAYOUT = nx.spring_layout(KG, seed=42, k=1.1)  


def classify_tumor_size(largest_dimension_mm: float) -> str:
    if largest_dimension_mm < 20:
        return "Small Tumor"
    elif largest_dimension_mm <= 40:
        return "Medium Tumor"
    return "Large Tumor"


def reason_over_graph(size_class: str) -> dict:
    symptoms = [u for u, _, d in KG.in_edges(size_class, data=True) if d["relation"] == "associated_with"]
    treatment = next((v for _, v, d in KG.out_edges(size_class, data=True) if d["relation"] == "requires_treatment"), None)
    outcome = None
    if treatment:
        outcome = next((v for _, v, d in KG.out_edges(treatment, data=True) if d["relation"] == "follows"), None)
    return {"symptoms": symptoms, "treatment": treatment, "outcome": outcome}


def render_knowledge_graph_png(highlighted_nodes) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=140)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    node_colors = ["#ff5a3c" if n in highlighted_nodes else "#3a4552" for n in KG.nodes]
    node_edge_colors = ["#ffb199" if n in highlighted_nodes else "#5b6774" for n in KG.nodes]

    nx.draw_networkx_edges(KG, _GRAPH_LAYOUT, edge_color="#5b6774", width=1.4, arrowsize=14, ax=ax)
    nx.draw_networkx_nodes(
        KG, _GRAPH_LAYOUT, node_color=node_colors, edgecolors=node_edge_colors,
        linewidths=1.6, node_size=1500, ax=ax,
    )
    nx.draw_networkx_labels(
        KG, _GRAPH_LAYOUT, font_size=7.6, font_color="#f4f1ea", font_family="monospace", ax=ax
    )
    edge_labels = {(u, v): d["relation"] for u, v, d in KG.edges(data=True)}
    nx.draw_networkx_edge_labels(
        KG, _GRAPH_LAYOUT, edge_labels=edge_labels, font_size=6.4,
        font_color="#9aa4b1", font_family="monospace", ax=ax,
    )
    ax.axis("off")

    buf = io.BytesIO()
    plt.tight_layout(pad=0.2)
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def preprocess(pil_img: Image.Image) -> torch.Tensor:
    # Matches the project's own Streamlit prototype: RGB, scale to [0,1], no resize.
    arr = np.array(pil_img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return tensor


def run_segmentation(pil_img: Image.Image) -> np.ndarray:
    tensor = preprocess(pil_img)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    binary_256 = (prob > 0.5).astype(np.uint8)
    orig_w, orig_h = pil_img.size
    binary_full = cv2.resize(binary_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return binary_full

def determine_lobe(centroid, img_shape):
    h, w = img_shape[:2]
    x, y = centroid
    hemisphere = "Left Hemisphere" if x < w / 2 else "Right Hemisphere"
    if y < h / 3:
        region = "Frontal Lobe"
    elif y < 2 * h / 3:
        region = "Parietal Lobe"
    else:
        region = "Occipital / Temporal Lobe"
    return f"{region} ({hemisphere})"


def extract_tumors(binary_mask: np.ndarray, img_shape) -> list:
    mask = (binary_mask > 0).astype(np.uint8)
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    tumors = []
    for i in range(1, num_labels): 
        x, y, w, h, area_px = stats[i]
        if area_px < 4:  
            continue
        cx, cy = centroids[i]
        height_mm = h * PIXEL_SPACING_MM
        width_mm = w * PIXEL_SPACING_MM
        area_mm2 = area_px * (PIXEL_SPACING_MM ** 2)
        size_class = classify_tumor_size(max(height_mm, width_mm))
        tumors.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "centroid_px": (float(cx), float(cy)),
            "area_px": int(area_px),
            "area_mm2": float(area_mm2),
            "height_mm": float(height_mm),
            "width_mm": float(width_mm),
            "location": determine_lobe((cx, cy), img_shape),
            "size_class": size_class,
        })
    tumors.sort(key=lambda t: t["area_px"], reverse=True)
    return tumors


def encode_png(arr_uint8_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(arr_uint8_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "scan" not in request.files or request.files["scan"].filename == "":
        return jsonify(error="No file uploaded."), 400

    file = request.files["scan"]
    if not allowed_file(file.filename):
        return jsonify(error="Please upload a .tif, .png, or .jpg MRI slice."), 400

    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify(error="Could not read that file as an image."), 400

    img_array = np.array(pil_img)
    binary_mask = run_segmentation(pil_img)
    tumors = extract_tumors(binary_mask, img_array.shape)

    if tumors:
        overlay = (mark_boundaries(img_array, binary_mask, color=(1, 0.32, 0.24), mode="thick") * 255).astype(np.uint8)
    else:
        overlay = img_array.copy()

    for t in tumors:
        x, y, w, h = t["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 180, 60), 1)
        cx, cy = t["centroid_px"]
        cv2.drawMarker(overlay, (int(cx), int(cy)), (255, 230, 90), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

    result = {
        "resolution": f"{img_array.shape[1]} x {img_array.shape[0]} px",
        "original": encode_png(img_array),
        "mask": encode_png(np.stack([binary_mask * 255] * 3, axis=-1).astype(np.uint8)),
        "overlay": encode_png(overlay),
        "tumor_count": len(tumors),
        "tumors": [],
    }

    for i, t in enumerate(tumors):
        reasoning = reason_over_graph(t["size_class"])
        highlight = {t["size_class"]}
        if reasoning["treatment"]:
            highlight.add(reasoning["treatment"])
        if reasoning["outcome"]:
            highlight.add(reasoning["outcome"])
        highlight.update(reasoning["symptoms"])

        result["tumors"].append({
            "index": i + 1,
            "area_mm2": round(t["area_mm2"], 2),
            "height_mm": round(t["height_mm"], 2),
            "width_mm": round(t["width_mm"], 2),
            "centroid_px": [round(c, 1) for c in t["centroid_px"]],
            "location": t["location"],
            "size_class": t["size_class"],
            "associated_symptoms": reasoning["symptoms"],
            "suggested_treatment": reasoning["treatment"],
            "likely_outcome_pattern": reasoning["outcome"],
            "knowledge_graph_png": render_knowledge_graph_png(highlight),
        })

    return jsonify(result)


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "1") == "1"
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
