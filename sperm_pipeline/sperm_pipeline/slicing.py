"""
Utilities for tiled inference over large video frames.
"""

from typing import Iterator, Tuple

import numpy as np


def iter_slices(
    height: int,
    width: int,
    slice_size: int,
    overlap: float,
) -> Iterator[Tuple[int, int, int, int]]:
    """
    Yield full-frame coordinates for a deterministic overlapping tile grid.
    Coordinates are (x1, y1, x2, y2).
    """
    slice_size = int(slice_size)
    if slice_size <= 0 or (width <= slice_size and height <= slice_size):
        yield 0, 0, width, height
        return

    overlap = min(max(float(overlap), 0.0), 0.9)
    step = max(1, int(round(slice_size * (1.0 - overlap))))

    max_x = max(0, width - slice_size)
    max_y = max(0, height - slice_size)

    xs = list(range(0, max_x + 1, step)) or [0]
    ys = list(range(0, max_y + 1, step)) or [0]
    if xs[-1] != max_x:
        xs.append(max_x)
    if ys[-1] != max_y:
        ys.append(max_y)

    for y1 in ys:
        for x1 in xs:
            yield x1, y1, min(x1 + slice_size, width), min(y1 + slice_size, height)


def nms_xyxy(boxes, scores, iou_threshold: float) -> np.ndarray:
    """
    Class-agnostic NMS for xyxy boxes. Returns kept indices in score order.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.size == 0 or scores.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    threshold = float(iou_threshold)
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

        order = rest[iou <= threshold]

    return np.asarray(keep, dtype=np.int64)
