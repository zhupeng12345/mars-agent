"""ROI extraction, light tracking and contact-sheet rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

try:
    import cv2
except Exception:  # pragma: no cover - model-only installations can still import types
    cv2 = None

from .visual_types import ROISequence


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return inter / max(area_a + area_b - inter, 1e-8)


def associate_boxes(previous: Optional[Sequence[float]], current: Sequence[float], threshold: float = 0.15) -> tuple[Optional[List[float]], float]:
    """Greedy one-track association, returning the selected box and IoU."""
    if not current:
        return (list(previous) if previous is not None else None), 0.0
    if previous is None:
        return list(current[0]), 1.0
    scores = [box_iou(previous, item) for item in current]
    best = int(np.argmax(scores))
    return (list(current[best]), float(scores[best])) if scores[best] >= threshold else (list(current[best]), float(scores[best]))


def _crop(frame: np.ndarray, box: Sequence[float], size: tuple[int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, min(w - 1, x1)), max(1, min(w, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    roi = frame[y1:y2, x1:x2]
    if cv2 is None:
        return np.asarray(roi)
    return cv2.resize(roi, size, interpolation=cv2.INTER_AREA)


def extract_roi_sequence(
    frames: Sequence[np.ndarray],
    detections: Optional[Sequence[Sequence[Mapping[str, Any] | Sequence[float]]]] = None,
    *,
    roi_size: tuple[int, int] = (224, 224),
    source_id: str = "window",
    frame_ids: Optional[Sequence[int]] = None,
    quality_threshold: float = 0.15,
) -> tuple[ROISequence, dict[str, Any]]:
    """Extract a fixed-size ROI sequence and track one target across frames.

    Missing detections are represented by ``frame_mask=False`` and a held box
    (or a zero box for the first missing frame), so downstream models can keep a
    fixed ``T`` while routing on the missing mask.
    """
    if not frames:
        raise ValueError("frames cannot be empty")
    if cv2 is None:
        raise RuntimeError("OpenCV is required for ROI extraction")
    t = len(frames)
    if detections is not None and len(detections) != t:
        raise ValueError("detections length must equal frames length")
    all_boxes: List[List[float]] = []
    confidence: List[float] = []
    rois: List[np.ndarray] = []
    frame_mask: List[bool] = []
    quality_mask: List[bool] = []
    previous: Optional[List[float]] = None
    previous_iou = 0.0
    for i, frame in enumerate(frames):
        entries = [] if detections is None else list(detections[i])
        normalized: List[tuple[List[float], float]] = []
        for item in entries:
            if isinstance(item, Mapping):
                box = item.get("bbox", item.get("box"))
                score = float(item.get("confidence", item.get("score", 1.0)))
            else:
                box, score = item, 1.0
            if box is not None and len(box) == 4:
                normalized.append((list(map(float, box)), score))
        selected, iou = associate_boxes(previous, [box for box, _ in normalized], threshold=quality_threshold)
        selected_score = 0.0
        if selected is not None:
            for box, score in normalized:
                if box == selected:
                    selected_score = score
                    break
        present = selected is not None and selected_score > 0
        if selected is None:
            selected = previous or [0.0, 0.0, 0.0, 0.0]
        if present:
            previous = list(selected)
            previous_iou = iou
        all_boxes.append(list(selected))
        confidence.append(float(selected_score))
        frame_mask.append(bool(present))
        quality_mask.append(bool(present and (i == 0 or previous_iou >= quality_threshold)))
        rois.append(_crop(frame, selected, roi_size))
    tensor = torch.from_numpy(np.stack(rois).transpose(0, 3, 1, 2)).float() / 255.0
    sequence = ROISequence.from_frame_tensor(
        tensor, boxes=torch.tensor(all_boxes, dtype=torch.float32).unsqueeze(0),
        frame_mask=torch.tensor(frame_mask).unsqueeze(0), quality_mask=torch.tensor(quality_mask).unsqueeze(0),
        source_ids=[source_id], weak_label=None,
        frame_ids=torch.tensor([list(frame_ids) if frame_ids is not None else list(range(t))], dtype=torch.long),
    )
    info = {
        "boxes": all_boxes, "confidences": confidence, "frame_mask": frame_mask,
        "quality_mask": quality_mask, "track_quality": float(np.mean(quality_mask)),
        "missing_count": int(sum(not value for value in frame_mask)),
        "key_frame_ids": [int(np.argmax(confidence))] if any(confidence) else [],
    }
    return sequence, info


def render_contact_sheet(
    frames: Sequence[np.ndarray], boxes: Sequence[Sequence[float]], rois: Sequence[np.ndarray],
    frame_ids: Optional[Sequence[int]] = None, output_path: str | Path = "contact_sheet.jpg",
    confidences: Optional[Sequence[float]] = None, frame_mask: Optional[Sequence[bool]] = None,
) -> str:
    """Write a two-row contact sheet with source overlays and aligned ROIs."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required for contact sheets")
    if not frames or len(frames) != len(boxes) or len(frames) != len(rois):
        raise ValueError("frames, boxes and rois must be non-empty and have equal length")
    tiles: List[np.ndarray] = []
    for i, (frame, box, roi) in enumerate(zip(frames, boxes, rois)):
        overlay = frame.copy()
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        present = True if frame_mask is None else bool(frame_mask[i])
        if present and x2 > x1 and y2 > y1:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 255), 3)
        label = f"frame {frame_ids[i] if frame_ids is not None else i}"
        if not present:
            label += "  missing"
        if confidences is not None:
            label += f"  conf={float(confidences[i]):.2f}"
        cv2.putText(overlay, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        overlay = cv2.resize(overlay, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
        tiles.append(np.hstack([overlay, roi]))
    sheet = np.vstack(tiles)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise IOError(f"unable to write contact sheet: {output}")
    return str(output)


__all__ = ["ROISequence", "box_iou", "associate_boxes", "extract_roi_sequence", "render_contact_sheet"]
