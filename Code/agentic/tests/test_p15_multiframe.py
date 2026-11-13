from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from agentic.models.fgmvae_temporal import FGMVAETemporal
from agentic.models.temporal import TemporalEncoder
from agentic.models.visual_types import ROISequence
from agentic.agents.visual_evidence import run_visual_evidence


class P15TemporalTests(unittest.TestCase):
    def test_all_temporal_profiles_and_masks(self) -> None:
        torch.manual_seed(7)
        x = torch.rand(2, 4, 5)
        mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
        for profile in ("conv", "gru", "tcn", "transformer"):
            model = TemporalEncoder(5, 8, profile).eval()
            output = model(x, mask)
            self.assertEqual(tuple(output.shape), (2, 8))
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.equal(output[1], torch.zeros_like(output[1])))

    def test_fgmvae_cpu_forward_loss_and_grad(self) -> None:
        torch.manual_seed(3)
        sequence = ROISequence.from_frame_tensor(
            torch.rand(2, 4, 3, 32, 32),
            boxes=torch.ones(2, 4, 4),
            frame_mask=torch.tensor([[True, True, True, False], [True, False, False, False]]),
            quality_mask=torch.ones(2, 4, dtype=torch.bool),
            source_ids=["a", "b"],
        )
        model = FGMVAETemporal(temporal_type="gru", mixture_components=4)
        outputs = model(sequence, sample=False)
        self.assertTrue(torch.allclose(outputs["mixture_weights"].sum(-1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.isfinite(outputs["kl"]))
        self.assertTrue(torch.isfinite(outputs["uncertainty"]).all())
        no_guide = FGMVAETemporal(temporal_type="gru", expert_dim=4)(sequence, sample=False)
        self.assertTrue(torch.isfinite(no_guide["kl"]))
        loss = model.loss(outputs, labels=torch.tensor([0, 2]))["total"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_single_frame_checkpoint_pools_video_frame_probabilities(self) -> None:
        model = FGMVAETemporal(model_profile="image-supervised-test")
        sequence = ROISequence.from_frame_tensor(torch.rand(1, 4, 3, 32, 32), source_ids=["video-test"])
        evidence = run_visual_evidence(model, sequence, frame_ids=[10, 20, 30, 40], run_id="run-framewise", thread_id="thread-framewise", framewise_pooling=True)
        self.assertEqual(evidence.metadata["input_mode"], "framewise_probability_pooling")
        self.assertTrue(evidence.key_frame_ids)
        self.assertAlmostEqual(sum(evidence.class_probs.values()), 1.0, places=5)

    def test_fuzzy_expert_guide_conditions_mixture_prior(self) -> None:
        torch.manual_seed(11)
        sequence = ROISequence.from_frame_tensor(torch.rand(2, 3, 3, 32, 32), source_ids=["guide-a", "guide-b"])
        model = FGMVAETemporal(temporal_type="gru", expert_dim=4, mixture_components=3)
        expert = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        output = model(sequence, expert_features=expert, sample=False)
        self.assertIsNotNone(output["feature_guide"])
        self.assertEqual(tuple(output["feature_guide"].shape), (2, 3))
        self.assertTrue(torch.allclose(output["feature_guide"].sum(-1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.isfinite(output["feature_guide"]).all())
        loss = output["mixture_weights"].square().mean() + output["kl"]
        loss.backward()
        generator_grads = [parameter.grad for parameter in model.feature_label_generator.parameters()]
        self.assertTrue(any(grad is not None and torch.isfinite(grad).all() for grad in generator_grads))
        shifted = model(sequence, expert_features=expert.flip(0), sample=False)["mixture_weights"]
        self.assertFalse(torch.allclose(output["mixture_weights"], shifted))

    def test_direct_feature_guide_is_not_discarded(self) -> None:
        torch.manual_seed(19)
        sequence = ROISequence.from_frame_tensor(torch.rand(1, 3, 3, 32, 32), source_ids=["direct-guide"])
        model = FGMVAETemporal(temporal_type="gru", mixture_components=3).eval()
        guide = torch.tensor([[0.96, 0.03, 0.01]])
        output = model(sequence, feature_guide=guide, sample=False)
        self.assertTrue(torch.allclose(output["feature_guide"], guide, atol=1e-6))
        self.assertGreater(float(output["mixture_weights"][0, 0].detach()), float(output["mixture_weights"][0, 1].detach()))
        with self.assertRaisesRegex(ValueError, "feature_guide must have shape"):
            model(sequence, feature_guide=torch.ones(1, 4), sample=False)
        with self.assertRaisesRegex(ValueError, "positive mass"):
            model(sequence, feature_guide=torch.zeros(1, 3), sample=False)

    def test_visual_agent_wires_static_dynamic_descriptors_into_fgmvae_t(self) -> None:
        torch.manual_seed(23)
        model = FGMVAETemporal(temporal_type="gru", expert_dim=9, mixture_components=3)
        sequence = ROISequence.from_frame_tensor(torch.rand(1, 4, 3, 32, 32), source_ids=["expert-agent"])
        evidence = run_visual_evidence(model, sequence, frame_ids=[1, 2, 3, 4], run_id="run-expert-agent", thread_id="thread-expert-agent")
        self.assertEqual(len(evidence.metadata["expert_feature_names"]), 9)
        self.assertEqual(evidence.metadata["expert_guidance"], "static_dynamic_fuzzy_prior")
        self.assertTrue(evidence.static_features)
        self.assertTrue(evidence.dynamic_features)

    def test_visual_agent_rejects_unregistered_expert_dimension(self) -> None:
        model = FGMVAETemporal(expert_dim=4)
        sequence = ROISequence.from_frame_tensor(torch.rand(1, 2, 3, 32, 32), source_ids=["bad-expert-dim"])
        with self.assertRaisesRegex(ValueError, "registered static/dynamic descriptors"):
            run_visual_evidence(model, sequence, run_id="run-bad-expert-dim", thread_id="thread-bad-expert-dim")


class P15ROIAndManifestTests(unittest.TestCase):
    def test_roi_sequence_keeps_missing_mask(self) -> None:
        cv2 = __import__("cv2")
        from agentic.models.roi_sequence import extract_roi_sequence, render_contact_sheet

        frames = [np.full((48, 64, 3), 20 + i * 5, dtype=np.uint8) for i in range(4)]
        detections = [[{"bbox": [8 + i, 8, 32 + i, 32], "confidence": 0.8}] if i != 2 else [] for i in range(4)]
        sequence, info = extract_roi_sequence(frames, detections, source_id="synthetic")
        self.assertEqual(tuple(sequence.frames.shape), (1, 4, 3, 224, 224))
        self.assertEqual(info["missing_count"], 1)
        self.assertFalse(bool(sequence.frame_mask[0, 2]))
        with tempfile.TemporaryDirectory() as directory:
            path = render_contact_sheet(frames, info["boxes"], [np.zeros((224, 224, 3), dtype=np.uint8)] * 4, output_path=Path(directory) / "sheet.jpg")
            self.assertTrue(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
