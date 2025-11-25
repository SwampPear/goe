
# heuristically or random sample 20%

"""
{
  'volume': FloatTensor[C, D, H, W],      # CT patch or full volume
  'paths': list[Tensor[Pi, 3]],          # each path = sequence of (z, y, x) points
  'path_mask': Optional[Tensor[1, D, H, W]],  # optional rasterized supervision
  'metadata': {
      'volume_id': str,
      'bbox': (z0, y0, x0, z1, y1, x1),  # if you’re using patches
  },
}
"""


class VesuviusSegmentationDataset:
    """
    Vesuvius challenge dataset for supervised segmentation.
    """
    def __init__(self):
        pass