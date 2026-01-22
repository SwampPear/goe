
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

"""
data_root/
  index.csv                   # global index over all training examples

  volumes/
    vol_000__volume.npy       # float32 [D, H, W] or [C, D, H, W]
    vol_001__volume.npy
    ...

  paths/
    vol_000__paths.npz        # np.savez(..., paths=[arr(N0,3), arr(N1,3), ...])
    vol_001__paths.npz
    ...

  masks/                      # optional (if you pre-rasterize)
    vol_000__pathmask.npy     # uint8/bool [D, H, W]
    vol_001__pathmask.npy
    ...

  meta/
    vol_000__meta.json        # spacing, origin, whatever
    vol_001__meta.json

"""


"""
All mapping data stored in .obj file



most from scroll 1 and 4
"""

class VesuviusSegmentationDownloader:
    """
    Downloads and processes raw volumetric and path data from the Vesuvius challenge data server.
    """
    def __init__(self):
        pass

    def download(self):
        # for artifacts needing downloading
        # download raw files
        # process them






class VesuviusSegmentationDataset:
    """
    Vesuvius challenge dataset indexer for supervised segmentation.
    """
    def __init__(self):
        pass