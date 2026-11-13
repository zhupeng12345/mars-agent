"""Build auditable multi-frame visual windows and ROI artifacts.

Examples:
    python -m agentic.scripts.build_visual_windows --output Code/data/evaluation/visual_manifest.jsonl

The script keeps video IDs grouped and records ``weak_source_label`` separately
from ``human_label``.  It can run without Ultralytics (for an offline smoke
test), in which case windows carry an explicit missing-mask/upgrade flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from agentic.models.roi_sequence import extract_roi_sequence, render_contact_sheet
from agentic.media.normalize import read_image


def _label_from_stem(stem: str) -> Optional[str]:
    if "偏高" in stem or stem.lower() in {"hot", "h"}:
        return "H"
    if "正常" in stem or stem.lower() in {"normal", "n"}:
        return "N"
    if "偏低" in stem or stem.lower() in {"cold", "c"}:
        return "C"
    return None


def _safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("_") or "media"


class YOLODetector:
    def __init__(self, weight: Optional[Path], device: str = "cpu") -> None:
        self.model = None
        self.backend = "unavailable"
        self.device = device
        if weight is None:
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(weight))
            self.backend = "yolo"
            names = self.model.names
            self.eye_id = next((int(k) for k, v in names.items() if str(v).lower() == "fireeye"), 0)
            self.fire_id = next((int(k) for k, v in names.items() if str(v).lower() == "fire"), 1)
        except Exception:
            self.model = None

    def __call__(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.model is None:
            return []
        try:
            result = self.model.predict(frame, imgsz=640, conf=0.15, device=self.device, verbose=False)[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return []
            xyxy = boxes.xyxy.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            conf = boxes.conf.detach().cpu().numpy()
            out = []
            for box, class_id, score in zip(xyxy, cls, conf):
                if class_id in (self.eye_id, self.fire_id):
                    out.append({"bbox": box.tolist(), "confidence": float(score), "class_id": int(class_id)})
            return out
        except Exception:
            return []


def _read_indices(path: Path, indices: Sequence[int]) -> tuple[List[np.ndarray], float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return [], 0.0, 0
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    wanted = set(int(x) for x in indices)
    frames: List[np.ndarray] = []
    i = 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if i in wanted:
            frames.append(frame)
            wanted.remove(i)
        i += 1
    capture.release()
    # A codec can return an out-of-order gap; preserve requested order.
    if len(frames) != len(indices):
        capture = cv2.VideoCapture(str(path))
        by_index = {}
        for target in sorted(set(indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(target))
            ok, frame = capture.read()
            if ok:
                by_index[int(target)] = frame
        capture.release()
        frames = [by_index[i] for i in indices if i in by_index]
    return frames, fps, total


def _video_records(
    path: Path, artifact_dir: Path, detector: YOLODetector, *, window_size: int, stride: int,
    sample_fps: float, max_windows: Optional[int], version: str,
) -> List[Dict[str, Any]]:
    probe = cv2.VideoCapture(str(path))
    if not probe.isOpened():
        return [{"source": "video", "video_path": str(path), "read_error": "decoder_open_failed", "label_status": "unlabeled/weak"}]
    fps = float(probe.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    probe.release()
    step = max(1, int(round(fps / max(sample_fps, 0.1))))
    sampled = list(range(0, total, step))
    if len(sampled) < window_size:
        sampled = list(range(min(total, window_size)))
    starts = list(range(0, max(1, len(sampled) - window_size + 1), max(1, stride)))
    if max_windows is not None:
        starts = starts[:max_windows]
    records = []
    weak_label = _label_from_stem(path.stem)
    # Chinese filenames are meaningful weak labels but collapse to the same
    # ASCII fallback; append a stable digest so the two source videos cannot
    # share a split group or overwrite artifacts.
    stem_tag = _label_from_stem(path.stem) or _safe_name(path.stem)
    media_id = f"video-{stem_tag.lower()}-{hashlib.sha1(str(path.resolve()).encode('utf-8', 'replace')).hexdigest()[:8]}"
    for window_index, start in enumerate(starts):
        frame_ids = sampled[start:start + window_size]
        if len(frame_ids) < window_size:
            continue
        frames, actual_fps, actual_total = _read_indices(path, frame_ids)
        if len(frames) != window_size:
            continue
        detections = [detector(frame) for frame in frames]
        sequence, track = extract_roi_sequence(frames, detections, source_id=media_id, frame_ids=frame_ids)
        window_id = f"{media_id}-w{window_index:04d}"
        window_dir = artifact_dir / media_id / window_id
        window_dir.mkdir(parents=True, exist_ok=True)
        frame_paths, roi_paths, overlay_paths = [], [], []
        for local_index, (frame, roi, box) in enumerate(zip(frames, sequence.frames[0], track["boxes"])):
            frame_path = window_dir / f"frame_{frame_ids[local_index]:06d}.jpg"
            roi_path = window_dir / f"roi_{local_index:02d}.jpg"
            overlay_path = window_dir / f"overlay_{local_index:02d}.jpg"
            overlay = frame.copy()
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            if track["frame_mask"][local_index]:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 255), 3)
            cv2.putText(overlay, f"frame={frame_ids[local_index]}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(str(frame_path), frame)
            cv2.imwrite(str(roi_path), cv2.cvtColor((roi.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(overlay_path), overlay)
            frame_paths.append(str(frame_path)); roi_paths.append(str(roi_path)); overlay_paths.append(str(overlay_path))
        contact_sheet = render_contact_sheet(frames, track["boxes"], [read_image(p) for p in roi_paths], frame_ids, window_dir / "contact_sheet.jpg", track["confidences"], track["frame_mask"])
        records.append({
            "schema_version": "visual-window-1", "implementation": "agentic.scripts.build_visual_windows", "version": version,
            "source": "video", "source_id": media_id, "media_id": media_id, "video_id": media_id,
            "video_path": str(path), "window_id": window_id, "window_index": window_index,
            "frame_paths": frame_paths, "roi_paths": roi_paths, "overlay_paths": overlay_paths,
            "frame_ids": frame_ids, "frame_start": frame_ids[0], "frame_end": frame_ids[-1],
            "timestamp_start": round(frame_ids[0] / max(actual_fps, 1e-6), 4), "timestamp_end": round(frame_ids[-1] / max(actual_fps, 1e-6), 4),
            "fps": round(actual_fps, 4), "total_frames": actual_total or total, "width": width, "height": height,
            "weak_source_label": weak_label, "human_label": None, "label_status": "unlabeled/weak",
            "split_group": media_id, "split": "unlabeled", "detection_backend": detector.backend,
            "boxes": track["boxes"], "confidences": track["confidences"], "frame_mask": track["frame_mask"],
            "quality_mask": track["quality_mask"], "missing_count": track["missing_count"],
            "track_quality": track["track_quality"], "key_frame_ids": track["key_frame_ids"],
            "contact_sheet": contact_sheet, "route_upgrade": bool(track["missing_count"] or detector.backend != "yolo"),
        })
    return records


def _image_records(image_dir: Path, version: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
            continue
        try:
            image = read_image(path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("decoder_returned_no_frame")
            h, w = image.shape[:2]
            valid = True
            error = None
        except Exception as exc:
            h = w = None; valid = False; error = f"decode_error:{type(exc).__name__}"
        label = _label_from_stem(path.parent.name)
        source_id = f"image-dir-{_safe_name(path.parent.name)}-{hashlib.sha1(str(path.parent.resolve()).encode('utf-8', 'replace')).hexdigest()[:8]}"
        image_id = f"image-{_safe_name(path.stem)}-{hashlib.sha1(str(path.resolve()).encode('utf-8', 'replace')).hexdigest()[:8]}"
        records.append({
            "schema_version": "visual-window-1", "implementation": "agentic.scripts.build_visual_windows", "version": version,
            "source": "image_directory", "source_id": source_id, "media_id": image_id,
            "window_id": image_id, "frame_paths": [str(path)], "frame_ids": [0],
            "frame_start": 0, "frame_end": 0, "width": w, "height": h, "read_error": error,
            "weak_source_label": label, "human_label": None, "label_status": "unlabeled/weak",
            "split_group": source_id, "split": "unlabeled", "frame_mask": [valid], "quality_mask": [valid],
            "boxes": [], "confidences": [], "route_upgrade": not valid,
        })
    return records


def build_manifest(
    video_dir: Path, image_dir: Optional[Path], output: Path, artifact_dir: Path,
    *, window_size: int = 8, stride: int = 4, sample_fps: float = 2.0,
    max_windows: Optional[int] = None, yolo_weight: Optional[Path] = None,
    device: str = "cpu", version: str = "p1.5-v1",
) -> List[Dict[str, Any]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    detector = YOLODetector(yolo_weight, device=device)
    records: List[Dict[str, Any]] = []
    for video in sorted(video_dir.glob("*")):
        if video.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            records.extend(_video_records(video, artifact_dir, detector, window_size=window_size, stride=stride, sample_fps=sample_fps, max_windows=max_windows, version=version))
    if image_dir is not None and image_dir.exists():
        records.extend(_image_records(image_dir, version))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    return records


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", default="Code/data/video")
    parser.add_argument("--image-dir", default="Code/data/火眼分类图片帧")
    parser.add_argument("--output", default="Code/data/evaluation/visual_manifest.jsonl")
    parser.add_argument("--artifact-dir", default="Code/data/evaluation/visual_windows")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--yolo-weight", default="Code/weights/best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--version", default="p1.5-v1")
    parser.add_argument("--no-yolo", action="store_true")
    args = parser.parse_args(argv)
    records = build_manifest(Path(args.video_dir), Path(args.image_dir) if args.image_dir else None, Path(args.output), Path(args.artifact_dir), window_size=args.window_size, stride=args.stride, sample_fps=args.sample_fps, max_windows=args.max_windows, yolo_weight=None if args.no_yolo else Path(args.yolo_weight), device=args.device, version=args.version)
    video_windows = [r for r in records if r.get("source") == "video" and r.get("frame_paths")]
    print(json.dumps({"records": len(records), "video_windows": len(video_windows), "video_ids": sorted({r.get("video_id") for r in video_windows}), "output": args.output}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
