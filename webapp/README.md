# Glioma Reasoning Console

A small Flask website for `neurosymbolic-ai-lgg-segmentation`: upload an MRI
slice, run it through the project's trained FPN segmentation model, and see
tumor geometry plus a symbolic knowledge-graph explanation (tumor size class
→ associated pattern → suggested treatment → outcome pattern).

This is a coursework / research prototype, **not a diagnostic tool.**

## 1. Where to put this folder

Drop this `webapp/` folder into the root of your cloned
`neurosymbolic-ai-lgg-segmentation` repo, next to `model/`:

```
neurosymbolic-ai-lgg-segmentation/
├── model/
│   ├── fpn_segmentation_weights.pth
│   └── ...
└── webapp/          <- this folder
    ├── app.py
    ├── requirements.txt
    ├── templates/index.html
    └── static/{style.css, script.js}
```

`app.py` looks for the weights at `../model/fpn_segmentation_weights.pth`
relative to itself. If you keep the weights somewhere else, set the
`MODEL_PATH` environment variable instead:

```bash
export MODEL_PATH=/absolute/path/to/fpn_segmentation_weights.pth
```

## 2. Install dependencies

```bash
cd webapp
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run it

```bash
python app.py
```

Open **http://127.0.0.1:5000** and drop in an MRI slice
(`model/sample_mri.tif` is a ready-made test image).

## What it does

1. **Neural stage** — the uploaded slice is run through the same
   `FPN_Segmentation` architecture trained in
   `model/fpn-segmentation-lgg.ipynb`, producing a binary tumor mask.
2. Connected-component analysis (same approach as `model/size_location.py`)
   turns the mask into per-region size (mm², assuming 0.5 mm/px spacing),
   height/width, centroid, and a quadrant-based brain lobe estimate.
3. **Symbolic stage** — each region's size class (small/medium/large) is
   looked up in a small `networkx` knowledge graph (a lightweight,
   Neo4j-free stand-in for `knowledge_graph.py`) to surface an associated
   symptom pattern, a suggested treatment, and a typical outcome pattern,
   with the matched path drawn on the graph.

## Notes / things to adjust for real use

- **Pixel spacing** is assumed at 0.5 mm/px (`PIXEL_SPACING_MM` in
  `app.py`), matching the assumption in `size_location.py`. Real scans
  carry this in DICOM metadata — wire that in if you move beyond `.tif`
  test slices.
- **The knowledge graph is illustrative**, trimmed from the project's
  prototype so the site doesn't need a running Neo4j instance. Swap
  `build_knowledge_graph()` in `app.py` for a call into your own graph
  (or into `knowledge_graph.py`'s Neo4j-backed version) once that
  service is running.
- Lobe/hemisphere localization is a simple quadrant heuristic on a single
  axial slice, not a registered brain atlas.
