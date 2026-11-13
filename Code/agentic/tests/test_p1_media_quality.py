from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - dependency is declared by agentic requirements
    cv2 = None

from agentic.agents.image_quality import ImageQualityAgent, evaluate_image_quality
from agentic.media.normalize import MediaNormalizer, build_media_manifest


@unittest.skipUnless(cv2 is not None, "OpenCV is required for media quality tests")
class P1MediaQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_image(self, name: str, image: np.ndarray) -> Path:
        path = self.root / name
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    def test_image_manifest_and_quality_gate(self) -> None:
        image = np.zeros((256, 320, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 40), (260, 210), (150, 180, 220), -1)
        cv2.circle(image, (160, 120), 35, (255, 220, 60), -1)
        path = self._write_image("clear.jpg", image)
        manifest = MediaNormalizer().normalize(path, run_id="run-test")
        self.assertEqual(manifest["media_type"], "image")
        self.assertEqual((manifest["width"], manifest["height"]), (320, 256))
        self.assertEqual(manifest["frame_count"], 1)
        self.assertEqual(len(manifest["sha256"]), 64)
        self.assertEqual(manifest["hash"], manifest["sha256"])
        self.assertIsNone(manifest["read_error"])
        result = evaluate_image_quality(manifest)
        self.assertIn(result["quality_status"], {"pass", "human"})
        self.assertTrue(result["target_present"])
        self.assertIn("cross_frame_consistency", result)

    def test_unicode_image_path_is_read_by_manifest_quality_and_visual_sequence(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(30, 220, size=(256, 320, 3), dtype=np.uint8)
        cv2.rectangle(image, (60, 45), (260, 210), (210, 170, 120), -1)
        path = self.root / "火眼图像_现场帧.jpg"
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        path.write_bytes(encoded.tobytes())

        manifest = MediaNormalizer().normalize(path, run_id="unicode-image-run")
        quality = ImageQualityAgent().run(manifest)
        from agentic.scripts.demo_run import _sequence_from_image

        sequence = _sequence_from_image(path, manifest["source_id"])
        self.assertEqual(manifest["read_error"], None)
        self.assertEqual((manifest["width"], manifest["height"]), (320, 256))
        self.assertIn(quality["quality_status"], {"pass", "human"})
        self.assertTrue(quality["target_present"])
        self.assertIsNotNone(sequence)
        self.assertEqual(sequence.source_ids, [manifest["source_id"]])

    def test_bad_quality_samples_route_to_retake(self) -> None:
        black = self._write_image("black.png", np.zeros((256, 256, 3), dtype=np.uint8))
        black_result = ImageQualityAgent().run(black)
        self.assertEqual(black_result["quality_status"], "retake")
        self.assertIn("target_missing", black_result["issues"])

        white = self._write_image("white.png", np.full((256, 256, 3), 255, dtype=np.uint8))
        white_result = ImageQualityAgent().run(white)
        self.assertEqual(white_result["quality_status"], "retake")
        self.assertIn("overexposure", white_result["issues"])

        missing_result = ImageQualityAgent().run(self.root / "missing.jpg")
        self.assertEqual(missing_result["quality_status"], "retake")
        self.assertIn("path_not_found", missing_result["issues"])

    def test_frame_directory_and_generated_video(self) -> None:
        frame_dir = self.root / "frames"
        frame_dir.mkdir()
        for index in range(4):
            image = np.full((120, 160, 3), 80 + index * 20, dtype=np.uint8)
            cv2.line(image, (5, 5), (150, 110), (240, 240, 240), 3)
            self.assertTrue(cv2.imwrite(str(frame_dir / f"frame_{index:03d}.jpg"), image))
        directory_manifest = MediaNormalizer().normalize(frame_dir, run_id="run-test")
        self.assertEqual(directory_manifest["media_type"], "frame_directory")
        self.assertEqual(directory_manifest["frame_count"], 4)
        directory_result = ImageQualityAgent().run(directory_manifest)
        self.assertEqual(directory_result["frame_count_evaluated"], 4)
        self.assertIsNotNone(directory_result["cross_frame_consistency"])

        video_path = self.root / "clip.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (160, 120))
        self.assertTrue(writer.isOpened())
        for index in range(6):
            frame = np.full((120, 160, 3), 90 + index * 10, dtype=np.uint8)
            cv2.rectangle(frame, (20 + index, 20), (120 + index, 90), (30, 180, 240), -1)
            writer.write(frame)
        writer.release()
        video_manifest = MediaNormalizer().normalize(video_path, run_id="run-test")
        self.assertEqual(video_manifest["media_type"], "video")
        self.assertEqual(video_manifest["width"], 160)
        self.assertEqual(video_manifest["height"], 120)
        self.assertEqual(video_manifest["frame_count"], 6)
        self.assertIsNone(video_manifest["read_error"])
        video_result = ImageQualityAgent().run(video_manifest)
        self.assertGreaterEqual(video_result["frame_count_evaluated"], 2)

    def test_build_multiple_manifests_keeps_duplicate_hashes(self) -> None:
        image = np.full((80, 80, 3), 100, dtype=np.uint8)
        first = self._write_image("first.png", image)
        second = self._write_image("second.png", image)
        manifests = build_media_manifest([first, second], run_id="run-test")
        self.assertEqual(len(manifests), 2)
        self.assertEqual(manifests[0]["sha256"], manifests[1]["sha256"])
        self.assertEqual(manifests[0]["media_id"], manifests[1]["media_id"])
        self.assertFalse(manifests[0]["metadata"]["duplicate_content"])
        self.assertTrue(manifests[1]["metadata"]["duplicate_content"])
        self.assertEqual(manifests[1]["metadata"]["duplicate_of_index"], 0)

    def test_oversized_media_is_rejected_before_hashing_or_decoding(self) -> None:
        path = self.root / "oversized.mp4"
        with path.open("wb") as handle:
            handle.truncate(2048)
        normalizer = MediaNormalizer(max_media_bytes=1024)
        with patch("agentic.media.normalize._sha256_file", side_effect=AssertionError("must not read oversized file")):
            manifest = normalizer.normalize(path, run_id="run-test")
        self.assertEqual(manifest["media_type"], "video")
        self.assertEqual(manifest["read_error"], "media_too_large")
        self.assertEqual(manifest["size_bytes"], 2048)
        self.assertEqual(manifest["sha256"], "")
        self.assertEqual(manifest["metadata"]["content_hash_status"], "not_computed_size_limit")
        quality = ImageQualityAgent().run(manifest)
        self.assertEqual(quality["quality_status"], "retake")

    def test_unsupported_extension_is_explicitly_rejected(self) -> None:
        path = self.root / "notes.txt"
        path.write_text("not an image or video", encoding="utf-8")
        manifest = MediaNormalizer().normalize(path)
        self.assertEqual(manifest["media_type"], "unknown")
        self.assertEqual(manifest["read_error"], "unsupported_media_type")
        self.assertEqual(manifest["metadata"]["error_code"], "unsupported_media_type")

    def test_corrupted_video_is_preserved_as_decode_error(self) -> None:
        path = self.root / "corrupt.mp4"
        path.write_bytes(b"not a valid mp4 container")
        manifest = MediaNormalizer().normalize(path)
        self.assertEqual(manifest["media_type"], "video")
        self.assertTrue(str(manifest["read_error"]).startswith("decode_error:"))
        self.assertEqual(manifest["metadata"]["error_code"], "decode_error")


if __name__ == "__main__":
    unittest.main()
