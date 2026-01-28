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