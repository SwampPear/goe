# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graph-of-Experts for the Vesuvius Challenge Kaggle competition: 3D papyrus surface segmentation in CT volumes of carbonized Herculaneum scrolls. The task is topology-aware 3D surface tracing (not generic segmentation). Output is binary voxel surface masks evaluated by a hybrid score (30% TopoScore + 35% SurfaceDice@τ + 35% VOI).

## Build & Run

**Package manager:** uv (Python ≥3.11, PyTorch 2.9.1+)

```bash
uv sync                  # install dependencies
uv run pytest             # run all tests
uv run pytest src/tests/test_shapes.py  # run a single test file
```

**Training & evaluation (scripts are stubs being rebuilt, reference commands from README):**
```bash
python scripts/train.py --data-root <DATA_ROOT> --split train --limit-volumes 1 --max-steps 5 --tiny
python scripts/eval.py --data-root <DATA_ROOT> --split val --limit-volumes 1 --tiny
```

## Architecture

The model is a **graph-structured mixture-of-experts** for 3D volumetric segmentation:

```
Input Volume → Stem (Conv3D stack) → Token Embeddings
  → Encoder (Transformer with positional encoding + cross-attention aux refinement)
  → Router (learned soft routing probabilities + adjacency matrix)
  → Experts (graph message-passing refinement with per-expert MLPs)
  → Decoder (expert context fusion + per-token MLP → volumetric logits)
```

**Key concepts:**
- Spatial patches become tokens that are dynamically routed to specialized experts
- Experts communicate via a learned adjacency matrix (graph message passing)
- Router produces load-balancing and entropy regularization losses to prevent expert collapse
- Top-K sparse routing (K=2-3) for efficiency; adaptive early stopping per token

## Code Layout

- `src/data/dataset.py` — Complete dataset pipeline: volume loading (tif/npy/zarr/nrrd), patch extraction, augmentations, custom collation. This is the most complete file in the new codebase.
- `src/models/` — Model stubs (`goe.py`, `components.py`, `routing.py`, `experts.py`, `refinement.py`) being rebuilt from the reference implementation.
- `src/old/models/` — **Fully implemented reference architecture** (`stem.py`, `encoder.py`, `router.py`, `experts.py`, `decoder.py`, `goe.py`). Use these as the source of truth when implementing the new model files.
- `src/training/` — Training loop, losses, metrics (stubs).
- `scripts/` — Entry points for train/eval (stubs; `scripts/old/` has working reference versions).
- `docs/architecture.md` — Detailed architecture design notes.
- `notes.md` — Planned optimizations (sparse activation, hierarchical attention, coarse-to-fine processing).

## Refactoring Status

The project is mid-refactor from `src/old/` (complete but unoptimized) to a clean `src/` structure. The `src/old/` directory contains the working reference implementation. New files in `src/models/`, `src/training/`, and `scripts/` are stubs that need to be filled in, incorporating optimizations from `notes.md`.

## Data Format

- Volumes are loaded via CSV index (`id,volume_path,split[,surface_path,spacing_z,spacing_y,spacing_x]`) or auto-discovered from directory layout (`data_root/{train|val|test}/<image_id>/volume.tif`)
- Dataset produces overlapping 3D patches with configurable `patch_size` and `stride`
- Supports multiple array formats: `.tif`, `.npy`, `.npz`, `.zarr`, `.nrrd`

## Competition Constraints

- Notebook-only submission, no internet, ≤9h runtime
- Output: one `<image_id>.tif` per test volume (exact shape + dtype match)
- Topology preservation is critical: avoid sheet merges/splits/spurious holes
