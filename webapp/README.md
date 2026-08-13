# Glioma Reasoning Console

A small Flask website for `neurosymbolic-ai-lgg-segmentation`: upload an MRI
slice, run it through the project's trained FPN segmentation model, and see
tumor geometry plus a symbolic knowledge-graph explanation (tumor size class
→ associated pattern → suggested treatment → outcome pattern).

This is a coursework research prototype

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
