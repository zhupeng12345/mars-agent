"""Provider-neutral LLM gateway with structured output and auditable metadata.

The gateway deliberately owns HTTP, credentials, retries and JSON decoding.
Agent nodes receive a stable response object and never need to duplicate
provider-specific request logic.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Type
from uuid import uuid4

import requests


DEFAULT_PROFILES = Path(__file__).resolve().parents[2] / "config" / "agent_profiles.json"


class GatewayError(RuntimeError):
    """Base class for provider and structured-output failures."""


class GatewayConfigurationError(GatewayError):
    """Raised when a profile cannot be used with the current environment."""


class GatewayProviderError(GatewayError):
    """Raised when a provider request fails or returns an invalid response."""


class GatewaySchemaError(GatewayError):
    """Raised when a provider response violates the requested output contract."""


@dataclass
class GatewayResponse:
    status: str
    profile: str
    provider: str
    model: str
    request_id: str
    content: Any = None
    raw_text: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    retries: int = 0
    fallback_profile: Optional[str] = None
    error_type: Optional[str] = None
    error: Optional[str] = None
    capabilities: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "profile": self.profile,
            "provider": self.provider,
            "model": self.model,
            "request_id": self.request_id,
            "content": self.content,
            "raw_text": self.raw_text,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "retries": self.retries,
            "fallback_profile": self.fallback_profile,
            "error_type": self.error_type,
            "error": self.error,
            "capabilities": self.capabilities,
        }


def _json_from_text(text: str) -> Any:
    text = str(text or "").strip()
    if not text:
        raise GatewayProviderError("provider_empty_response")
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    start = text.find("{")
    if start >= 0:
        candidates.append(text[start:])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise GatewayProviderError("provider_non_json_response")


def _image_payload(image: Any) -> str:
    if isinstance(image, str):
        if image.startswith("data:"):
            return image
        path = Path(image)
        if path.is_file():
            return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        return image
    if isinstance(image, (bytes, bytearray)):
        return "data:image/jpeg;base64," + base64.b64encode(bytes(image)).decode("ascii")
    raise TypeError("image must be bytes, path or data URL")


def _schema_validate(payload: Any, response_schema: Any) -> Any:
    if response_schema is None:
        return payload
    if isinstance(payload, Mapping) and hasattr(response_schema, "from_dict"):
        value = response_schema.from_dict(dict(payload))
    elif isinstance(payload, Mapping) and callable(response_schema):
        value = response_schema(**dict(payload))
    else:
        value = payload
    validator = getattr(value, "validate", None)
    if callable(validator):
        validator()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(value)
    return value


def _provider_json_schema(response_schema: Any) -> Optional[Dict[str, Any]]:
    schema_factory = getattr(response_schema, "json_schema", None)
    if callable(schema_factory):
        schema = schema_factory()
        if isinstance(schema, Mapping):
            return dict(schema)
    return None


def _schema_name(response_schema: Any) -> str:
    name = getattr(response_schema, "__name__", type(response_schema).__name__)
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name)).strip("_").lower()
    return normalized or "structured_response"


class LLMGateway:
    """Single entry point for local, cloud and fake providers."""

    def __init__(self, profiles_path: Optional[Path] = None, session: Optional[requests.Session] = None):
        self.profiles_path = Path(profiles_path or DEFAULT_PROFILES)
        self.session = session or requests.Session()
        # Local Ollama calls should not inherit a system HTTP proxy, which can
        # be malformed or unavailable even while localhost itself is healthy.
        # Keep the injected session for deterministic transport tests.
        self.ollama_session = session or requests.Session()
        if session is None:
            self.ollama_session.trust_env = False

    def profiles(self) -> Dict[str, Dict[str, Any]]:
        if not self.profiles_path.exists():
            raise GatewayConfigurationError(f"profile_file_not_found:{self.profiles_path}")
        data = json.loads(self.profiles_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise GatewayConfigurationError("profile_file_must_be_object")
        return {str(k): dict(v) for k, v in data.items() if isinstance(v, Mapping)}

    def _resolve(self, profile: str) -> Dict[str, Any]:
        profiles = self.profiles()
        if profile not in profiles:
            raise GatewayConfigurationError(f"unknown_profile:{profile}")
        config = dict(profiles[profile])
        config["profile"] = profile
        config.setdefault("capabilities", [])
        config.setdefault("timeout", 60)
        config.setdefault("max_tokens", 1024)
        config.setdefault("temperature", 0.1)
        return config

    @staticmethod
    def _check_config(config: Mapping[str, Any], images: Sequence[Any], response_schema: Any = None) -> Optional[str]:
        provider = str(config.get("provider", ""))
        if provider in {"fake", "fallback"}:
            return None
        if not str(config.get("base_url", "")).strip():
            return "missing_base_url"
        if not str(config.get("model", "")).strip():
            return "missing_model"
        if provider == "openai_compatible":
            output_mode = str(config.get("structured_output_mode", "json_object"))
            if output_mode not in {"json_object", "json_schema"}:
                return f"unsupported_structured_output_mode:{output_mode}"
            if output_mode == "json_schema" and _provider_json_schema(response_schema) is None:
                return "json_schema_mode_requires_response_schema"
            env_name = str(config.get("api_key_env", "MARS_LLM_API_KEY"))
            if not os.getenv(env_name, "").strip() and not os.getenv("MARS_LLM_API_KEY", "").strip():
                return f"missing_api_key:{env_name}"
        if images and "vision" not in set(config.get("capabilities", [])):
            return "profile_missing_vision_capability"
        return None

    def chat(
        self,
        profile: str,
        messages: Iterable[Mapping[str, Any]],
        images: Optional[Sequence[Any]] = None,
        response_schema: Any = None,
        trace_context: Optional[Mapping[str, Any]] = None,
    ) -> GatewayResponse:
        config = self._resolve(profile)
        images = list(images or [])
        request_id = f"req-{uuid4().hex[:12]}"
        started = time.perf_counter()
        problem = self._check_config(config, images, response_schema)
        if problem:
            return GatewayResponse(
                status="configuration_error",
                profile=profile,
                provider=str(config.get("provider", "unknown")),
                model=str(config.get("model", "")),
                request_id=request_id,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                error_type="configuration_error",
                error=problem,
                capabilities=list(config.get("capabilities", [])),
            )
        provider = str(config.get("provider"))
        if provider == "fake":
            value: Any = {
                "visual_summary": "fake provider structured response",
                "visual_evidence": ["fake_evidence"],
                "knowledge_evidence": [],
                "risk_interpretation": "unknown",
                "uncertainties": ["fake provider"],
                "follow_up_checks": ["human review"],
            }
            fake_payload = getattr(response_schema, "fake_payload", None)
            if callable(fake_payload):
                value = fake_payload()
            try:
                value = _schema_validate(value, response_schema)
            except Exception as exc:
                return self._error_response(config, request_id, started, "schema_error", str(exc))
            return self._success_response(config, request_id, started, value, "", 0)
        if provider == "fallback":
            return GatewayResponse(
                status="fallback_required",
                profile=profile,
                provider=provider,
                model=str(config.get("model", "human-review")),
                request_id=request_id,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                fallback_profile="human_review",
                capabilities=list(config.get("capabilities", [])),
            )

        retries = max(0, int(config.get("retries", 1)))
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                if provider == "ollama":
                    raw_text, usage = self._call_ollama(config, list(messages), images, response_schema)
                else:
                    raw_text, usage = self._call_openai_compatible(config, list(messages), images, response_schema)
                parsed = _json_from_text(raw_text)
                try:
                    value = _schema_validate(parsed, response_schema)
                except (ValueError, TypeError, KeyError) as exc:
                    raise GatewaySchemaError(f"response_schema_validation_failed:{exc}") from exc
                return self._success_response(config, request_id, started, value, raw_text, attempt, usage)
            except GatewaySchemaError as exc:
                return self._error_response(config, request_id, started, "schema_error", str(exc), attempt)
            except (requests.RequestException, ValueError, TypeError, GatewayError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(min(0.25 * (attempt + 1), 1.0))
        return self._error_response(config, request_id, started, type(last_error).__name__, str(last_error), retries)

    def _success_response(self, config: Mapping[str, Any], request_id: str, started: float, content: Any, raw_text: str, retries: int, usage: Optional[Dict[str, Any]] = None) -> GatewayResponse:
        return GatewayResponse(
            status="completed",
            profile=str(config["profile"]),
            provider=str(config["provider"]),
            model=str(config.get("model", "")),
            request_id=request_id,
            content=content,
            raw_text=raw_text,
            usage=dict(usage or {}),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            retries=retries,
            capabilities=list(config.get("capabilities", [])),
        )

    def _error_response(self, config: Mapping[str, Any], request_id: str, started: float, error_type: str, error: str, retries: int = 0) -> GatewayResponse:
        return GatewayResponse(
            status="schema_error" if error_type == "schema_error" else "provider_error",
            profile=str(config["profile"]),
            provider=str(config["provider"]),
            model=str(config.get("model", "")),
            request_id=request_id,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            retries=retries,
            error_type=error_type,
            error=error,
            capabilities=list(config.get("capabilities", [])),
        )

    def _call_ollama(self, config: Mapping[str, Any], messages: list[Mapping[str, Any]], images: Sequence[Any], response_schema: Any = None) -> tuple[str, Dict[str, Any]]:
        payload_messages = [dict(message) for message in messages]
        if images:
            for message in payload_messages:
                if message.get("role") == "user":
                    message["images"] = [self._ollama_image(image) for image in images]
                    break
        schema = _provider_json_schema(response_schema)
        payload = {"model": config["model"], "messages": payload_messages, "format": schema or "json", "stream": False, "options": {"temperature": config["temperature"], "num_predict": config["max_tokens"]}}
        if "think" in config:
            payload["think"] = bool(config["think"])
        response = self.ollama_session.post(
            str(config["base_url"]).rstrip("/") + "/api/chat",
            json=payload,
            timeout=float(config["timeout"]),
        )
        response.raise_for_status()
        body = response.json()
        message = body.get("message", {}) if isinstance(body, Mapping) else {}
        usage = {}
        if isinstance(body, Mapping):
            for source, target in (("prompt_eval_count", "prompt_tokens"), ("eval_count", "completion_tokens"), ("total_duration", "total_duration_ns"), ("prompt_eval_duration", "prompt_duration_ns"), ("eval_duration", "completion_duration_ns")):
                if body.get(source) is not None:
                    usage[target] = body[source]
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                usage["total_tokens"] = int(usage["prompt_tokens"]) + int(usage["completion_tokens"])
        return str(message.get("content", "")), usage

    @staticmethod
    def _ollama_image(image: Any) -> str:
        payload = _image_payload(image)
        return payload.split(",", 1)[1] if payload.startswith("data:") else payload

    def _call_openai_compatible(self, config: Mapping[str, Any], messages: list[Mapping[str, Any]], images: Sequence[Any], response_schema: Any = None) -> tuple[str, Dict[str, Any]]:
        prepared = []
        image_blocks = [{"type": "image_url", "image_url": {"url": _image_payload(image)}} for image in images]
        for message in messages:
            item = dict(message)
            if image_blocks and item.get("role") == "user" and isinstance(item.get("content"), str):
                item["content"] = [{"type": "text", "text": item["content"]}, *image_blocks]
            prepared.append(item)
        env_name = str(config.get("api_key_env", "MARS_LLM_API_KEY"))
        api_key = os.getenv(env_name, "") or os.getenv("MARS_LLM_API_KEY", "")
        output_mode = str(config.get("structured_output_mode", "json_object"))
        if output_mode not in {"json_object", "json_schema"}:
            raise GatewayConfigurationError(f"unsupported_structured_output_mode:{output_mode}")
        request_format: Dict[str, Any] = {"type": "json_object"}
        if output_mode == "json_schema":
            schema = _provider_json_schema(response_schema)
            if schema is None:
                raise GatewayConfigurationError("json_schema_mode_requires_response_schema")
            request_format = {
                "type": "json_schema",
                "json_schema": {"name": _schema_name(response_schema), "strict": True, "schema": schema},
            }
        response = self.session.post(
            str(config["base_url"]).rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": config["model"], "messages": prepared, "temperature": config["temperature"], "max_tokens": config["max_tokens"], "response_format": request_format},
            timeout=float(config["timeout"]),
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices", []) if isinstance(body, Mapping) else []
        if not choices:
            raise GatewayProviderError("provider_empty_choices")
        choice = choices[0]
        message = choice.get("message", {}) if isinstance(choice, Mapping) else {}
        if message.get("refusal"):
            raise GatewayProviderError("provider_refusal")
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            raise GatewayProviderError("provider_truncated_response")
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping) and part.get("type") in {"text", "output_text"})
        elif content is None:
            content = ""
        return str(content), dict(body.get("usage", {}) or {})
