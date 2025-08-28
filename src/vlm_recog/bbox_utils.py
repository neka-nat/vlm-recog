"""Utility functions for bounding boxes and segmentation masks"""

def area(box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    inter = intersection(a, b)
    if inter == 0:
        return 0.0
    return inter / (area(a) + area(b) - inter)


def coverage(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection / Smaller area"""
    inter = intersection(a, b)
    if inter == 0:
        return 0.0
    return inter / min(area(a), area(b))


def add_margin(box: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = box
    return xmin - margin, ymin - margin, xmax + margin, ymax + margin


def clip_box(box: tuple[int, int, int, int], W: int, H: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(W, x1))
    y1 = max(0, min(H, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    return x1, y1, x2, y2


def merge_boxes(
    boxes: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
    coverage_threshold: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        merged_here = False

        for i, m in enumerate(merged):
            if (
                iou(box, m) >= iou_threshold
                or coverage(box, m) >= coverage_threshold
            ):
                mx1, my1, mx2, my2 = m
                merged[i] = (
                    min(xmin, mx1),
                    min(ymin, my1),
                    max(xmax, mx2),
                    max(ymax, my2),
                )
                merged_here = True
                break

        if not merged_here:
            merged.append(box)

    # Take care of overlapping boxes
    if len(merged) < len(boxes):
        return merge_boxes(merged, iou_threshold, coverage_threshold)

    return merged
