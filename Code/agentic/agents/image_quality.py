"""Explainable image and sampled-video quality checks.

The agent is intentionally model-free.  Its output is suitable as a quality
gate before expensive visual diagnosis and contains only observable metrics;
it never emits private model reasoning or process-control advice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - minimal installs can still inspect errors
    cv2 = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

from agentic.media.normalize import IMAGE_EXTENSIONS, MediaNormalizer, read_image


@dataclass(frozen=True)
class QualityThresholds:
    """Versioned thresholds used by :class:`ImageQualityAgent`."""

    version: str = "image-quality-1.0"
    blur_pass: float = 0.18
    blur_human: float = 0.10
    brightness_pass: float = 0.25
    brightness_human: float = 0.12
    overexposure_pass: float = 0.08
    overexposure_human: float = 0.25
    occlusion_pass: float = 0.70
    occlusion_human: float = 0.45
    resolution_pass: float = 0.65
    resolution_human: float = 0.35
    cross_frame_pass: float = 0.55
    cross_frame_human: float = 0.30


def _read_image(path: Path) -> Optional[np.ndarray]:
    if cv2 is not None:
        image = read_image(path, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    if Image is not None:
        try:
            with Image.open(path) as handle:
                rgb = np.asarray(handle.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if cv2 is not None else rgb[:, :, ::-1]
        except Exception:
            return None
    return None


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    if cv2 is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.mean(image[..., :3], axis=2).astype(np.float32)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _frame_metrics(image: np.ndarray) -> Dict[str, float]:
    gray = _gray(image)
    height, width = gray.shape[:2]
    mean = float(np.mean(gray)) if gray.size else 0.0
    std = float(np.std(gray)) if gray.size else 0.0
    if cv2 is not None and gray.size:
        variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        edge_density = float(np.count_nonzero(edges) / edges.size) if edges.size else 0.0
    else:
        variance = float(np.var(np.diff(gray, axis=0))) if gray.shape[0] > 1 else 0.0
        edge_density = 0.0
    # Scale the Laplacian into [0, 1] while retaining enough separation between
    # blurred and sharp industrial frames.  Raw variance is included in output.
    blur_score = min(1.0, max(0.0, variance / 500.0))
    brightness_score = max(0.0, 1.0 - abs(mean - 127.5) / 127.5)
    overexposure_ratio = float(np.mean(gray >= 250.0)) if gray.size else 1.0

    border = max(1, int(min(height, width) * 0.05)) if height and width else 1
    border_values = np.concatenate([
        gray[:border, :].ravel(), gray[-border:, :].ravel(),
        gray[:, :border].ravel(), gray[:, -border:].ravel(),
    ]) if gray.size else np.asarray([], dtype=np.float32)
    if border_values.size:
        blocked = float(np.mean((border_values <= 8.0) | (border_values >= 247.0)))
        # Black borders are common around furnace/fire-eye imagery and do not by
        # themselves mean that the target is occluded.  Penalize them strongly
        # only when the whole frame is flat or when both border and center are
        # covered; the target-presence check handles a genuinely blank frame.
        center = gray[border:-border, border:-border] if min(height, width) > 2 * border else gray
        center_std = float(np.std(center)) if center.size else 0.0
        if blocked > 0.95 and center_std > 8.0:
            occlusion_score = 0.75
        else:
            occlusion_score = max(0.0, min(1.0, 1.0 - blocked))
    else:
        occlusion_score = 0.0
    min_dimension = min(height, width) if height and width else 0
    resolution_score = min(1.0, min_dimension / 256.0)
    target_present = bool(std >= 8.0 and (edge_density >= 0.002 or std >= 18.0))
    return {
        "blur_score": blur_score,
        "blur_raw_variance": variance,
        "brightness_score": brightness_score,
        "brightness_mean": mean / 255.0,
        "overexposure_ratio": overexposure_ratio,
        "occlusion_score": occlusion_score,
        "resolution_score": resolution_score,
        "edge_density": edge_density,
        "target_present": float(target_present),
        "width": float(width),
        "height": float(height),
    }


def _aggregate(metrics: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    result: Dict[str, float] = {}
    numeric_names = [
        "blur_score", "blur_raw_variance", "brightness_score", "brightness_mean",
        "overexposure_ratio", "occlusion_score", "resolution_score", "edge_density",
        "width", "height",
    ]
    for name in numeric_names:
        values = [_safe_float(item.get(name)) for item in metrics if name in item]
        if values:
            result[name] = float(np.median(values))
    result["target_present"] = 1.0 if any(bool(item.get("target_present")) for item in metrics) else 0.0
    return result


class ImageQualityAgent:
    """Evaluate image, video and frame-directory quality before diagnosis."""

    agent_id = "image_quality"

    def __init__(self, *, thresholds: Optional[QualityThresholds] = None, max_frames: int = 12) -> None:
        self.thresholds = thresholds or QualityThresholds()
        self.max_frames = max(2, int(max_frames))
        self.normalizer = MediaNormalizer()

    def run(
        self,
        media: Union[str, Path, Mapping[str, Any]],
        *,
        manifest: Optional[Mapping[str, Any]] = None,
        run_id: str = "",
        thread_id: str = "",
    ) -> Dict[str, Any]:
        """Return the stable quality output schema.

        ``media`` may be a path or a manifest dictionary.  Passing both is
        useful when a caller already normalized the input and wants to preserve
        the same run/thread lineage.
        """
        if isinstance(media, Mapping):
            manifest_payload = dict(media)
            path = Path(str(manifest_payload.get("path", "")))
        else:
            path = Path(media)
            manifest_payload = dict(manifest or self.normalizer.normalize(path, run_id=run_id, thread_id=thread_id))
        if manifest is not None:
            manifest_payload = dict(manifest)
            path = Path(str(manifest_payload.get("path", path)))
        media_type = str(manifest_payload.get("media_type", ""))
        if not media_type:
            media_type = str(self.normalizer.normalize(path).get("media_type", "unknown"))

        frames, frame_errors = self._load_frames(path, media_type)
        metrics = [_frame_metrics(frame) for frame in frames]
        aggregate = _aggregate(metrics)
        cross_frame = self._cross_frame_score(frames)
        result = self._result_from_metrics(
            aggregate, cross_frame=cross_frame, manifest=manifest_payload,
            frame_count=len(frames), frame_errors=frame_errors,
        )
        result["run_id"] = run_id or manifest_payload.get("run_id", "")
        result["thread_id"] = thread_id or manifest_payload.get("thread_id", "")
        result["agent_id"] = self.agent_id
        result["implementation"] = "agentic.agents.image_quality"
        return result

    evaluate = run
    analyze = run

    def _load_frames(self, path: Path, media_type: str) -> Tuple[List[np.ndarray], List[str]]:
        errors: List[str] = []
        if not path.exists():
            return [], ["path_not_found"]
        if media_type == "image" or (path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
            frame = _read_image(path)
            return ([frame] if frame is not None else []), ([] if frame is not None else ["decode_error"])
        if media_type == "frame_directory" or path.is_dir():
            candidates = sorted(
                [item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS],
                key=lambda item: item.name.lower(),
            )
            if not candidates:
                return [], ["empty_frame_directory"]
            selected = self._sample_indices(len(candidates))
            frames: List[np.ndarray] = []
            for index in selected:
                frame = _read_image(candidates[index])
                if frame is None:
                    errors.append(f"decode_error:{candidates[index].name}")
                else:
                    frames.append(frame)
            return frames, errors
        if media_type == "video" or path.suffix.lower() not in IMAGE_EXTENSIONS:
            if cv2 is None:
                return [], ["decoder_unavailable"]
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    return [], ["decoder_open_failed"]
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                selected = self._sample_indices(count) if count > 0 else list(range(self.max_frames))
                frames: List[np.ndarray] = []
                for index in selected:
                    if count > 0:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        errors.append(f"decode_error:frame_{index}")
                    else:
                        frames.append(frame)
                return frames, errors
            finally:
                capture.release()
        return [], ["unsupported_media_type"]

    def _sample_indices(self, count: int) -> List[int]:
        if count <= 0:
            return []
        if count <= self.max_frames:
            return list(range(count))
        # Include first and last frame; evenly spaced samples avoid adjacent-only
        # bias for long videos and preserve a deterministic sequence.
        return sorted(set(int(round(value)) for value in np.linspace(0, count - 1, self.max_frames)))

    def _cross_frame_score(self, frames: Sequence[np.ndarray]) -> Optional[float]:
        if len(frames) < 2:
            return 1.0 if frames else None
        normalized: List[np.ndarray] = []
        for frame in frames:
            gray = _gray(frame)
            if cv2 is not None:
                gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            else:
                gray = gray[:64, :64]
            normalized.append(gray / 255.0)
        distances = [float(np.mean(np.abs(a - b))) for a, b in zip(normalized, normalized[1:])]
        return max(0.0, min(1.0, 1.0 - float(np.mean(distances)))) if distances else 1.0

    def _result_from_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        cross_frame: Optional[float],
        manifest: Mapping[str, Any],
        frame_count: int,
        frame_errors: Sequence[str],
    ) -> Dict[str, Any]:
        threshold = self.thresholds
        issues: List[str] = []
        read_error = manifest.get("read_error")
        if read_error:
            issues.append(str(read_error))
        if frame_errors:
            issues.extend(str(item) for item in frame_errors[:8])
        if not metrics:
            issues.append("no_decodable_frame")
            return {
                "quality_status": "retake" if read_error or frame_errors else "human",
                "blur_score": 0.0,
                "brightness_score": 0.0,
                "overexposure_ratio": 1.0,
                "occlusion_score": 0.0,
                "resolution_score": 0.0,
                "target_present": False,
                "cross_frame_consistency": None,
                "cross_frame_score": None,
                "issues": sorted(set(issues)),
                "evidence": [],
                "frame_count_evaluated": 0,
                "threshold_version": threshold.version,
                "manifest": dict(manifest),
            }

        blur = _safe_float(metrics.get("blur_score"))
        brightness = _safe_float(metrics.get("brightness_score"))
        overexposure = _safe_float(metrics.get("overexposure_ratio"))
        occlusion = _safe_float(metrics.get("occlusion_score"))
        resolution = _safe_float(metrics.get("resolution_score"))
        target_present = bool(metrics.get("target_present"))
        if blur < threshold.blur_human:
            issues.append("blur")
        elif blur < threshold.blur_pass:
            issues.append("blur_borderline")
        if brightness < threshold.brightness_human:
            issues.append("brightness")
        elif brightness < threshold.brightness_pass:
            issues.append("brightness_borderline")
        if overexposure > threshold.overexposure_human:
            issues.append("overexposure")
        elif overexposure > threshold.overexposure_pass:
            issues.append("overexposure_borderline")
        if occlusion < threshold.occlusion_human:
            issues.append("occlusion")
        elif occlusion < threshold.occlusion_pass:
            issues.append("occlusion_borderline")
        if resolution < threshold.resolution_human:
            issues.append("resolution")
        elif resolution < threshold.resolution_pass:
            issues.append("resolution_borderline")
        if not target_present:
            issues.append("target_missing")
        if cross_frame is not None:
            if cross_frame < threshold.cross_frame_human:
                issues.append("cross_frame_instability")
            elif cross_frame < threshold.cross_frame_pass:
                issues.append("cross_frame_borderline")

        severe = {"blur", "brightness", "overexposure", "occlusion", "resolution", "target_missing", "no_decodable_frame"}
        if read_error or frame_errors or severe.intersection(issues):
            status = "retake"
        elif issues:
            status = "human"
        else:
            status = "pass"
        evidence = [
            {"metric": "blur_score", "value": blur, "threshold": threshold.blur_pass},
            {"metric": "brightness_score", "value": brightness, "threshold": threshold.brightness_pass},
            {"metric": "overexposure_ratio", "value": overexposure, "threshold": threshold.overexposure_pass},
            {"metric": "occlusion_score", "value": occlusion, "threshold": threshold.occlusion_pass},
            {"metric": "resolution_score", "value": resolution, "threshold": threshold.resolution_pass},
            {"metric": "target_present", "value": target_present},
        ]
        if cross_frame is not None:
            evidence.append({"metric": "cross_frame_consistency", "value": cross_frame, "threshold": threshold.cross_frame_pass})
        return {
            "quality_status": status,
            "blur_score": blur,
            "brightness_score": brightness,
            "overexposure_ratio": overexposure,
            "occlusion_score": occlusion,
            "resolution_score": resolution,
            "target_present": target_present,
            "cross_frame_consistency": cross_frame,
            "cross_frame_score": cross_frame,
            "issues": sorted(set(issues)),
            "evidence": evidence,
            "frame_count_evaluated": frame_count,
            "threshold_version": threshold.version,
            "manifest": dict(manifest),
        }


def evaluate_image_quality(
    media: Union[str, Path, Mapping[str, Any]],
    *,
    manifest: Optional[Mapping[str, Any]] = None,
    thresholds: Optional[QualityThresholds] = None,
    max_frames: int = 12,
    run_id: str = "",
    thread_id: str = "",
) -> Dict[str, Any]:
    """Convenience API for callers that do not need to retain an agent object."""
    return ImageQualityAgent(thresholds=thresholds, max_frames=max_frames).run(
        media, manifest=manifest, run_id=run_id, thread_id=thread_id
    )


__all__ = ["ImageQualityAgent", "QualityThresholds", "evaluate_image_quality"]
