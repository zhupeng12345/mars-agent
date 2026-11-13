"""Normalize images, videos and extracted frame directories.

The normalizer is deliberately defensive: a manifest is returned even when a
path cannot be opened.  Consumers can route a manifest containing ``read_error``
to retake or human review without having to catch decoder-specific exceptions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

try:  # OpenCV is an optional dependency for callers that only inspect manifests.
    import cv2
except Exception:  # pragma: no cover - exercised only in minimal installations
    cv2 = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - exercised only in minimal installations
    Image = None

try:
    from agentic.platform.contracts import MediaManifest, MediaType
except Exception:  # pragma: no cover - allows direct module use from a checkout
    MediaManifest = None  # type: ignore[assignment,misc]

    class MediaType:  # type: ignore[no-redef]
        IMAGE = "image"
        VIDEO = "video"
        FRAME_DIRECTORY = "frame_directory"
        UNKNOWN = "unknown"


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"})


def _natural_key(path: Path) -> List[Union[int, str]]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def read_image(path: Union[str, Path], flags: Optional[int] = None):
    """Read an image path, including Unicode paths on Windows OpenCV builds."""
    if cv2 is None:
        return None
    flags = cv2.IMREAD_COLOR if flags is None else flags
    path_text = str(path)
    if os.name != "nt" or path_text.isascii():
        image = cv2.imread(path_text, flags)
        if image is not None:
            return image
    try:
        import numpy as np

        encoded = np.fromfile(path_text, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    return cv2.imdecode(encoded, flags) if encoded.size else None


def _safe_source_id(value: Optional[str], digest: str) -> str:
    if value:
        cleaned = re.sub(r"[^A-Za-z0-9_.:@/-]+", "_", str(value)).strip("_")
        if cleaned:
            return cleaned
    return f"media-{digest[:16]}"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(paths: Sequence[Path], root: Path) -> str:
    """Hash relative names and bytes so directory order and duplicates are stable."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=_natural_key):
        relative = path.relative_to(root).as_posix().encode("utf-8", "surrogatepass")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def _manifest_dict(
    *,
    media_id: str,
    source_id: str,
    path: Path,
    media_type: str,
    sha256: str,
    file_hash: Optional[str] = None,
    size_bytes: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
    duration_seconds: Optional[float] = None,
    frame_count: Optional[int] = None,
    read_error: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    run_id: str = "",
    thread_id: str = "",
) -> Dict[str, Any]:
    if size_bytes is None:
        try:
            if path.is_file():
                size_bytes = path.stat().st_size
            elif path.is_dir():
                size_bytes = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        except Exception:
            size_bytes = None
    values: Dict[str, Any] = {
        "media_id": media_id,
        "source_id": source_id,
        "path": str(path),
        "media_type": media_type,
        "sha256": sha256,
        "file_hash": file_hash or sha256,
        "size_bytes": size_bytes,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "frame_count": frame_count,
        "read_error": read_error,
        "metadata": dict(metadata or {}),
        "run_id": run_id,
        "thread_id": thread_id,
    }
    if MediaManifest is not None:
        try:
            # ``MediaManifest`` supplies schema/audit defaults and validates enums.
            payload = MediaManifest(**values).to_dict()
            # ``sha256`` is canonical; ``hash`` keeps the small manifest schema
            # convenient for callers that do not care which digest algorithm is
            # used by the normalizer.
            payload.setdefault("hash", sha256)
            return payload
        except Exception:
            # Keep a useful dict when an older contracts module is imported by a
            # caller.  The same field names remain compatible with the contract.
            pass
    values.update({
        "schema_version": "media-manifest-1",
        "agent_id": "input_normalizer",
        "implementation": "agentic.media.normalize",
        "hash": sha256,
    })
    return values


class MediaNormalizer:
    """Create deterministic manifests for one media path."""

    def __init__(self, *, max_video_probe_frames: int = 8, max_media_bytes: Optional[int] = 4 * 1024**3) -> None:
        self.max_video_probe_frames = max(1, int(max_video_probe_frames))
        if max_media_bytes is not None and (
            isinstance(max_media_bytes, bool) or not isinstance(max_media_bytes, int) or max_media_bytes <= 0
        ):
            raise ValueError("max_media_bytes must be a positive integer or None")
        self.max_media_bytes = max_media_bytes

    def normalize(
        self,
        path: Union[str, Path],
        *,
        source_id: Optional[str] = None,
        run_id: str = "",
        thread_id: str = "",
    ) -> Dict[str, Any]:
        path_obj = Path(path)
        try:
            resolved = path_obj.expanduser().resolve()
        except Exception:
            resolved = path_obj

        if not resolved.exists():
            digest = hashlib.sha256(str(resolved).encode("utf-8", "replace")).hexdigest()
            return _manifest_dict(
                media_id=f"media-{digest[:16]}", source_id=_safe_source_id(source_id, digest),
                path=resolved, media_type=MediaType.UNKNOWN.value, sha256=digest,
                read_error="path_not_found", metadata={"error_code": "path_not_found"},
                run_id=run_id, thread_id=thread_id,
            )

        size_bytes = self._path_size_bytes(resolved)
        if self.max_media_bytes is not None and size_bytes is not None and size_bytes > self.max_media_bytes:
            return self._oversized_manifest(
                resolved, size_bytes=size_bytes, source_id=source_id, run_id=run_id, thread_id=thread_id
            )

        if resolved.is_dir():
            return self._normalize_directory(resolved, source_id=source_id, run_id=run_id, thread_id=thread_id)
        suffix = resolved.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            return self._normalize_video(resolved, source_id=source_id, run_id=run_id, thread_id=thread_id)
        if suffix in IMAGE_EXTENSIONS:
            return self._normalize_image(resolved, source_id=source_id, run_id=run_id, thread_id=thread_id)
        digest = self._hash_or_fallback(resolved)
        return _manifest_dict(
            media_id=f"media-{digest[:16]}", source_id=_safe_source_id(source_id, digest),
            path=resolved, media_type=MediaType.UNKNOWN.value, sha256=digest,
            read_error="unsupported_media_type", metadata={"suffix": suffix, "error_code": "unsupported_media_type"},
            run_id=run_id, thread_id=thread_id,
        )

    @staticmethod
    def _path_size_bytes(path: Path) -> Optional[int]:
        try:
            if path.is_file():
                return path.stat().st_size
            if path.is_dir():
                return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        except OSError:
            return None
        return None

    def _oversized_manifest(
        self,
        path: Path,
        *,
        size_bytes: int,
        source_id: Optional[str],
        run_id: str,
        thread_id: str,
    ) -> Dict[str, Any]:
        try:
            modified_ns = path.stat().st_mtime_ns
        except OSError:
            modified_ns = None
        identity = f"{path}|{size_bytes}|{modified_ns}".encode("utf-8", "replace")
        identity_hash = hashlib.sha256(identity).hexdigest()
        suffix = path.suffix.lower()
        if path.is_dir():
            media_type = MediaType.FRAME_DIRECTORY.value
        elif suffix in VIDEO_EXTENSIONS:
            media_type = MediaType.VIDEO.value
        elif suffix in IMAGE_EXTENSIONS:
            media_type = MediaType.IMAGE.value
        else:
            media_type = MediaType.UNKNOWN.value
        return _manifest_dict(
            media_id=f"media-{identity_hash[:16]}",
            source_id=_safe_source_id(source_id, identity_hash),
            path=path,
            media_type=media_type,
            sha256="",
            file_hash="",
            size_bytes=size_bytes,
            read_error="media_too_large",
            metadata={
                "error_code": "media_too_large",
                "max_media_bytes": self.max_media_bytes,
                "content_hash_status": "not_computed_size_limit",
                "identity_fingerprint": identity_hash,
            },
            run_id=run_id,
            thread_id=thread_id,
        )

    def _hash_or_fallback(self, path: Path) -> str:
        try:
            return _sha256_file(path)
        except Exception:
            return hashlib.sha256(str(path).encode("utf-8", "replace")).hexdigest()

    def _normalize_image(self, path: Path, *, source_id: Optional[str], run_id: str, thread_id: str) -> Dict[str, Any]:
        digest = self._hash_or_fallback(path)
        width = height = None
        read_error = None
        metadata: Dict[str, Any] = {"format": path.suffix.lower().lstrip(".")}
        try:
            if Image is not None:
                with Image.open(path) as image:
                    width, height = image.size
                    metadata.update({"format": image.format or metadata["format"], "mode": image.mode})
            elif cv2 is not None:
                image = read_image(path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError("decoder_returned_no_frame")
                height, width = image.shape[:2]
            else:
                raise RuntimeError("no_image_decoder_available")
        except Exception as exc:
            read_error = f"decode_error:{type(exc).__name__}:{exc}"
            metadata["error_code"] = "decode_error"
        return _manifest_dict(
            media_id=f"media-{digest[:16]}", source_id=_safe_source_id(source_id, digest), path=path,
            media_type=MediaType.IMAGE.value, sha256=digest, width=width, height=height,
            frame_count=1 if read_error is None else None, read_error=read_error, metadata=metadata,
            run_id=run_id, thread_id=thread_id,
        )

    def _normalize_video(self, path: Path, *, source_id: Optional[str], run_id: str, thread_id: str) -> Dict[str, Any]:
        digest = self._hash_or_fallback(path)
        width = height = frame_count = None
        fps = duration = None
        read_error = None
        metadata: Dict[str, Any] = {"decoder": "opencv"}
        if cv2 is None:
            read_error = "decoder_unavailable:opencv"
            metadata["error_code"] = "decoder_unavailable"
        else:
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    raise ValueError("decoder_open_failed")
                width_value = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                height_value = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps_value = capture.get(cv2.CAP_PROP_FPS)
                count_value = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                width = int(width_value) if width_value and width_value > 0 else None
                height = int(height_value) if height_value and height_value > 0 else None
                fps = float(fps_value) if fps_value and fps_value > 0 else None
                frame_count = int(count_value) if count_value and count_value > 0 else None
                if fps and frame_count:
                    duration = float(frame_count / fps)
                # Probe a few frames so a file with a valid header but no decodable
                # payload is still routed to retake/human review.
                decoded = 0
                for _ in range(self.max_video_probe_frames):
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    decoded += 1
                metadata["probe_decoded_frames"] = decoded
                if decoded == 0:
                    raise ValueError("decoder_returned_no_frame")
                if frame_count is None:
                    frame_count = decoded
                    if fps:
                        duration = float(frame_count / fps)
            except Exception as exc:
                read_error = f"decode_error:{type(exc).__name__}:{exc}"
                metadata["error_code"] = "decode_error"
            finally:
                capture.release()
        return _manifest_dict(
            media_id=f"media-{digest[:16]}", source_id=_safe_source_id(source_id, digest), path=path,
            media_type=MediaType.VIDEO.value, sha256=digest, width=width, height=height, fps=fps,
            duration_seconds=duration, frame_count=frame_count, read_error=read_error, metadata=metadata,
            run_id=run_id, thread_id=thread_id,
        )

    def _normalize_directory(self, path: Path, *, source_id: Optional[str], run_id: str, thread_id: str) -> Dict[str, Any]:
        files = sorted((item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS), key=_natural_key)
        digest = _sha256_directory(files, path) if files else hashlib.sha256(str(path).encode("utf-8", "replace")).hexdigest()
        width = height = None
        invalid_files: List[str] = []
        for item in files:
            try:
                if Image is not None:
                    with Image.open(item) as image:
                        candidate_width, candidate_height = image.size
                elif cv2 is not None:
                    image = read_image(item, cv2.IMREAD_UNCHANGED)
                    if image is None:
                        raise ValueError("decoder_returned_no_frame")
                    candidate_height, candidate_width = image.shape[:2]
                else:
                    raise RuntimeError("no_image_decoder_available")
                if width is None:
                    width, height = int(candidate_width), int(candidate_height)
            except Exception:
                invalid_files.append(str(item.relative_to(path)))
        read_error = None
        if not files:
            read_error = "empty_frame_directory"
        elif invalid_files:
            read_error = f"decode_error:{len(invalid_files)}_frame_files"
        metadata: Dict[str, Any] = {
            "frame_files": len(files),
            "invalid_frame_files": invalid_files[:32],
            "recursive": True,
        }
        if invalid_files:
            metadata["error_code"] = "decode_error"
        if not invalid_files and files:
            metadata["frame_pattern"] = [item.relative_to(path).as_posix() for item in files[:3]]
        return _manifest_dict(
            media_id=f"media-{digest[:16]}", source_id=_safe_source_id(source_id, digest), path=path,
            media_type=MediaType.FRAME_DIRECTORY.value, sha256=digest, width=width, height=height,
            frame_count=len(files) if files else 0, read_error=read_error, metadata=metadata,
            run_id=run_id, thread_id=thread_id,
        )


def normalize_media(
    path: Union[str, Path],
    *,
    max_media_bytes: Optional[int] = 4 * 1024**3,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience wrapper returning a JSON-safe manifest dictionary."""
    return MediaNormalizer(max_media_bytes=max_media_bytes).normalize(path, **kwargs)


def build_media_manifest(
    paths: Union[Union[str, Path], Iterable[Union[str, Path]]],
    *,
    source_prefix: Optional[str] = None,
    run_id: str = "",
    thread_id: str = "",
    max_media_bytes: Optional[int] = 4 * 1024**3,
) -> List[Dict[str, Any]]:
    """Normalize one path or an iterable of paths into manifest records."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    normalizer = MediaNormalizer(max_media_bytes=max_media_bytes)
    result = []
    first_by_hash: Dict[str, tuple[int, str]] = {}
    for index, path in enumerate(paths):
        source_id = f"{source_prefix}-{index}" if source_prefix else None
        manifest = normalizer.normalize(path, source_id=source_id, run_id=run_id, thread_id=thread_id)
        digest = str(manifest.get("sha256", ""))
        if digest and digest in first_by_hash:
            first_index, first_path = first_by_hash[digest]
            manifest.setdefault("metadata", {}).update({
                "duplicate_content": True,
                "duplicate_of_index": first_index,
                "duplicate_of_path": first_path,
            })
        elif digest:
            first_by_hash[digest] = (index, str(path))
            manifest.setdefault("metadata", {})["duplicate_content"] = False
        result.append(manifest)
    return result


def manifests_to_jsonl(manifests: Iterable[Mapping[str, Any]], output: Union[str, Path]) -> None:
    """Write manifests as UTF-8 JSON Lines for dataset and audit tooling."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for manifest in manifests:
            handle.write(json.dumps(dict(manifest), ensure_ascii=False, sort_keys=True) + "\n")


__all__ = [
    "IMAGE_EXTENSIONS", "VIDEO_EXTENSIONS", "MediaNormalizer", "build_media_manifest",
    "manifests_to_jsonl", "normalize_media",
]
