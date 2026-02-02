# Architecture

## Input Stem
- converts volumetric CT scans into tokenized representations for surface reasoning
- 3D convolutional encoder extracts surface-relevant features (curvature gradients, layer boundaries, void patterns)
- auxiliary routing features encode local geometry cues to bias initial expert selection

### Key Optimizations
- **coarse-to-fine multi-scale processing:** overlapping voxel patches provide multi-resolution surface context

## Encoder
## Graph Router
## Experts and Refinement
## Decoder

# Metrics
- Surface IoU / Dice
- Inference FLOPs per volume
- Expert utilization entropy (is GoE actually using multiple experts?)