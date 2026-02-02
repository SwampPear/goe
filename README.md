Vesuvius Challenge – Surface Detection (Kaggle)

Task:
3D papyrus surface segmentation in CT volumes of carbonized scrolls.

Input:

3D CT scan chunks (variable size, voxel spacing given)

Output:

Binary voxel mask of papyrus sheet surface

Recto (horizontal fibers) preferred; approximate sheet OK

Key Constraints:

Preserve topology

Avoid sheet merges, splits, and spurious holes/handles

Evaluation (Final Score ∈ [0,1])
Score = 0.30 × TopoScore
      + 0.35 × SurfaceDice@τ
      + 0.35 × VOI_score


τ = 2.0 (physical units)

SurfaceDice@τ: surface proximity within tolerance
VOI_score: penalizes splits/merges (26-connectivity)
TopoScore: Betti-based topology (k=0 components, k=1 tunnels, k=2 cavities)

Voxel Dice alone is insufficient.

Dataset Notes

3D CT from ESRF (BM18) and DLS (I12)

Carbonized papyrus; damaged, compressed layers

More (less-curated) labels may be released mid-competition

Submission

submission.zip

One [image_id].tif per test volume

Exact shape + dtype match

Notebook submissions only

No internet; ≤9h runtime (CPU/GPU)

Core Insight:
This is topology-aware 3D surface tracing, not generic segmentation or ink detection.

Quickstart: Initial Test

Prereqs:
- Prepare a processed dataset with `index.csv` in your data root. The file should at minimum include:
  `id,volume_path,split` and optionally `surface_path` (preferred), `ink_path`, `geometry_path`.
  `volume_path` and `surface_path`/`ink_path` are relative to the data root.
- If you have voxel spacing, include `spacing_z,spacing_y,spacing_x` or a single `spacing` column.

Smoke-test commands (CPU):
1) Train a tiny model on a small slice
   python scripts/train.py --data-root <DATA_ROOT> --split train --limit-volumes 1 --max-steps 5 --tiny

2) Eval sanity metrics (Dice/IoU only, not leaderboard metrics)
   python scripts/eval.py --data-root <DATA_ROOT> --split val --limit-volumes 1 --tiny

3) Predict and create submission.zip (binary .tif per volume)
   python scripts/predict.py --data-root <DATA_ROOT> --split test --limit-volumes 1 --tiny --out submission.zip
