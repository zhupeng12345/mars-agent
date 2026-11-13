"""Typed boundaries for ROI windows.

The platform contracts contain the report-facing ``VisualEvidence`` object.
This module contains the tensor-facing object used before an Agent result is
created, so accidental ``B/T/C/H/W`` shape changes fail at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch


@dataclass
class ROISequence:
    """A batch of aligned ROI windows.

    ``frames`` is ``[B,T,3,H,W]`` and is expected to be float32 in ``[0,1]``
    (normalisation for a pretrained backbone happens in the model).  Boxes are
    pixel coordinates ``[x1,y1,x2,y2]`` in the source frame.
    """

    frames: torch.Tensor
    boxes: torch.Tensor
    frame_mask: torch.Tensor
    quality_mask: torch.Tensor
    source_ids: List[str] = field(default_factory=list)
    weak_label: Optional[str] = None
    frame_ids: Optional[torch.Tensor] = None

    def validate(self) -> "ROISequence":
        if not isinstance(self.frames, torch.Tensor) or self.frames.ndim != 5:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        b, t, c, h, w = self.frames.shape
        if c != 3 or h <= 0 or w <= 0:
            raise ValueError("frames must have shape [B,T,3,H,W] with positive spatial size")
        if not torch.isfinite(self.frames).all():
            raise ValueError("frames contain NaN or Inf")
        if self.frames.dtype not in (torch.float16, torch.float32, torch.float64):
            raise ValueError("frames must be floating point")
        if self.frames.min().item() < -1e-5 or self.frames.max().item() > 1.00001:
            raise ValueError("frames must be in [0,1]")
        if not isinstance(self.boxes, torch.Tensor) or self.boxes.shape != (b, t, 4):
            raise ValueError("boxes must have shape [B,T,4]")
        if not torch.isfinite(self.boxes).all() or (self.boxes < 0).any():
            raise ValueError("boxes must contain finite non-negative coordinates")
        for name, mask in (("frame_mask", self.frame_mask), ("quality_mask", self.quality_mask)):
            if not isinstance(mask, torch.Tensor) or mask.shape != (b, t):
                raise ValueError(f"{name} must have shape [B,T]")
            if mask.dtype != torch.bool:
                raise ValueError(f"{name} must be bool")
        if self.frame_ids is not None:
            if not isinstance(self.frame_ids, torch.Tensor) or self.frame_ids.shape != (b, t):
                raise ValueError("frame_ids must have shape [B,T]")
        if self.source_ids and len(self.source_ids) != b:
            raise ValueError("source_ids length must equal batch size")
        return self

    @property
    def effective_mask(self) -> torch.Tensor:
        return self.frame_mask & self.quality_mask

    @classmethod
    def from_frame_tensor(
        cls,
        frames: torch.Tensor,
        *,
        boxes: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
        quality_mask: Optional[torch.Tensor] = None,
        source_ids: Optional[Sequence[str]] = None,
        weak_label: Optional[str] = None,
        frame_ids: Optional[torch.Tensor] = None,
    ) -> "ROISequence":
        if frames.ndim == 4:
            frames = frames.unsqueeze(0)
        if frames.ndim != 5:
            raise ValueError("frames must have shape [B,T,3,H,W] or [T,3,H,W]")
        b, t = frames.shape[:2]
        boxes = torch.zeros((b, t, 4), dtype=torch.float32, device=frames.device) if boxes is None else boxes
        frame_mask = torch.ones((b, t), dtype=torch.bool, device=frames.device) if frame_mask is None else frame_mask
        quality_mask = torch.ones((b, t), dtype=torch.bool, device=frames.device) if quality_mask is None else quality_mask
        item = cls(frames.float(), boxes.float(), frame_mask.bool(), quality_mask.bool(), list(source_ids or []), weak_label, frame_ids)
        return item.validate()

    def to(self, device: torch.device | str) -> "ROISequence":
        return ROISequence(
            self.frames.to(device), self.boxes.to(device), self.frame_mask.to(device),
            self.quality_mask.to(device), list(self.source_ids), self.weak_label,
            None if self.frame_ids is None else self.frame_ids.to(device),
        )
