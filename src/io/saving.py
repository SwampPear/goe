import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import zlib


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_list_feature(value: np.ndarray) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.tolist()))


def write_tfrecord(
    features: np.ndarray,
    *,
    path: str | Path,
    scroll_id: int,
    intensity_range: Optional[Tuple[float, float]] = None,
    compression: Optional[str] = "GZIP",
) -> None:
    """
    Write a (Z,Y,X,C) float32 features array into a TFRecord file.

    Args:
        features: np.ndarray (Z,Y,X,C)
        path: output .tfrecord file path
        scroll_id: integer scroll identifier
        intensity_range: optional tuple (vmin,vmax)
        compression: 'GZIP', 'ZLIB', or None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    options = tf.io.TFRecordOptions(compression_type=compression or "")
    with tf.io.TFRecordWriter(str(path), options=options) as writer:
        # Serialize feature tensor (compressed to save space)
        raw = features.tobytes()
        compressed = zlib.compress(raw, level=5)

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "scroll_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[scroll_id])),
                    "shape": _int64_list_feature(np.array(features.shape, dtype=np.int64)),
                    "features": _bytes_feature(compressed),
                }
            )
        )
        if intensity_range:
            example.features.feature["intensity_min"].float_list.value.append(float(intensity_range[0]))
            example.features.feature["intensity_max"].float_list.value.append(float(intensity_range[1]))

        writer.write(example.SerializeToString())

    print(f"[OK] Saved TFRecord: {path} | shape={features.shape} | scroll={scroll_id}")
