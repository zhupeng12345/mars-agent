"""Optional HTTP adapter for an external JEV-compatible router."""

from __future__ import annotations

from typing import Any, Mapping

import requests


class JevHttpAdapter:
    def __init__(self, base_url: str, *, timeout: float = 5.0, session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def predict(self, features: Any) -> dict[str, Any]:
        payload = features.to_dict() if hasattr(features, "to_dict") else dict(features)
        response = self.session.post(self.base_url + "/route", json=payload, timeout=self.timeout)
        response.raise_for_status()
        body = response.json()
        return dict(body) if isinstance(body, Mapping) else {"abstain": True, "route": "human", "confidence": 0.0}
