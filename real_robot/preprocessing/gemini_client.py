"""Structured Gemini calls over uploaded videos, cached on disk.

Two backends, one interface:

* ``generate_content`` -- ``client.models.generate_content`` with
  ``types.VideoMetadata(fps=..., start_offset=..., end_offset=...)`` on each
  video part and a JSON schema on the response. Documented as legacy and fully
  supported; the default.
* ``interactions`` -- ``client.interactions.create`` with a video input whose
  ``processing`` sets the sampling rate, a per-video ``resolution``, a
  ``generation_config`` carrying the temperature and the output limit, and a
  ``response_format`` schema. Clips are cut locally and uploaded rather than
  addressed by offset.

A configured ``temperature`` is sent by both backends; ``null`` leaves the
model's default. Google's Interactions guide documents
``generation_config={"temperature": ...}``, although its API reference does not
list the field. On the general-use Gemini 3.x models temperature, top-p and
top-k are deprecated and ignored by either API, so there the setting changes
nothing and a request may differ from one run to the next -- the response
cache, not the temperature, is what makes a stage repeatable.

Sampling is always set explicitly. Gemini's default for static video is one
frame per second, which would reduce a 15 Hz episode to every fifteenth frame.

Callers order parts so that what is shared comes first -- the fixed
specification, then the videos, then the pass-specific text -- which lets
implicit prefix caching reuse the specification and the videos across passes.
Cached input tokens are reported in each call's normalised usage.

Every successful response is written to ``cache_dir`` under a hash of everything
that determines it -- backend, model, generation settings, prompt text, each
video's content hash, fps and clip bounds, and the schema -- so re-running a
stage costs nothing and a changed prompt can never be answered from an old
cache entry. Failures are handled by what caused them:

* a transient service error (408, 429, 5xx, timeouts) is retried with backoff;
* a response cut off at the output limit raises :class:`GeminiTruncated` at
  once -- asking again identically would be charged and cut off again, so the
  caller asks for less instead;
* malformed JSON is asked again at most ``malformed_json_retries`` times;
* anything else raises.

Every attempt is recorded with its outcome and token usage, and the text, usage
and finish reason of every failed response are saved under
``cache_dir/failures``, one file per attempt. Failed responses are billed like
successful ones, so the usage a call reports -- on its record, or on the
exception it raises -- is the sum over all of its attempts. The API key is read
from the environment and never written.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ..common import canonical_json, file_sha256, read_json, repo_path, utc_now, write_json

UPLOAD_REUSE_SECONDS = 44 * 3600    # files expire after 48 h
RETRIABLE_CODES = {408, 429, 500, 502, 503, 504}


USAGE_KEYS = ("input_tokens", "output_tokens", "cached_tokens", "thought_tokens")


class GeminiError(RuntimeError):
    """A call that produced no usable answer. ``attempts`` records every try and what it used."""

    def __init__(self, message: str, attempts: Optional[Sequence[Mapping[str, Any]]] = None):
        super().__init__(message)
        self.attempts: List[Dict[str, Any]] = [dict(a) for a in attempts or ()]

    @property
    def usage(self) -> Dict[str, int]:
        return sum_usage(a.get("usage_normalized") for a in self.attempts)


class GeminiRequestFailed(GeminiError):
    """The service refused or never answered, after retries: quota, network, a rejected request."""


class GeminiTruncated(GeminiError):
    """The response stopped at the output limit. Ask for less rather than the same again."""


class GeminiMalformed(GeminiError):
    """The response was not parseable JSON."""


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class VideoPart:
    """A prepared video, optionally clipped to ``start_frame..end_frame`` inclusive."""

    path: str
    fps: float
    source_fps: float
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None


Part = Union[TextPart, VideoPart]


def parse_json_text(text: str) -> Any:
    """JSON from a model reply, tolerating a Markdown fence around it."""
    if text is None:
        raise ValueError("empty response")
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    if fence:
        stripped = fence.group(1)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start:end + 1])
        raise


def _to_dict(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    for method in ("model_dump", "to_dict", "to_json_dict"):
        fn = getattr(value, method, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                continue
    if isinstance(value, Mapping):
        return {k: _to_dict(v) for k, v in value.items()}
    return str(value)


def normalize_usage(usage: Any) -> Dict[str, Optional[int]]:
    """Input, output, cached and thinking tokens from either backend's usage record."""
    usage = usage if isinstance(usage, Mapping) else {}

    def first(*names: str) -> Optional[int]:
        for name in names:
            value = usage.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
        return None

    return {
        "input_tokens": first("prompt_token_count", "total_input_tokens"),
        "output_tokens": first("candidates_token_count", "total_output_tokens"),
        "cached_tokens": first("cached_content_token_count", "total_cached_tokens"),
        "thought_tokens": first("thoughts_token_count", "total_thought_tokens"),
    }


def sum_usage(items: Iterable[Optional[Mapping[str, Any]]]) -> Dict[str, int]:
    """Token counts added over several normalised usage records; a missing count adds nothing."""
    total = {key: 0 for key in USAGE_KEYS}
    for item in items:
        for key in USAGE_KEYS:
            value = (item or {}).get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total[key] += int(value)
    return total


def is_truncated(finish: Optional[str]) -> bool:
    text = str(finish or "").upper()
    return "MAX_TOKENS" in text or text.endswith("INCOMPLETE")


class GeminiClient:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = dict(cfg)
        self.backend = str(cfg["backend"])
        if self.backend not in ("generate_content", "interactions"):
            raise ValueError(f"gemini.backend must be generate_content or interactions, got {self.backend!r}")
        self.model = str(cfg["model"])
        self.cache_dir = repo_path(cfg["cache_dir"])
        self._client = None
        self.use_response_cache = True
        self._model_info = None

    # ----------------------------------------------------------- client
    @property
    def client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise GeminiError("google-genai is not installed; see requirements-preprocess.txt") from exc
            key = os.environ.get(str(self.cfg.get("api_key_env", "GEMINI_API_KEY"))) or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise GeminiError(
                    f"no API key: export {self.cfg.get('api_key_env', 'GEMINI_API_KEY')} in the shell "
                    "that runs this stage"
                )
            self._client = genai.Client(api_key=key)
        return self._client

    def settings(self) -> Dict[str, Any]:
        """Everything besides the parts and schema that shapes a response -- exactly what is sent."""
        return {
            "backend": self.backend,
            "model": self.model,
            "temperature": None if self.cfg.get("temperature") is None else float(self.cfg["temperature"]),
            "max_output_tokens": int(self.cfg.get("max_output_tokens", 65536)),
            "media_resolution": str(self.cfg.get("media_resolution", "default")),
        }

    # ---------------------------------------------------------- uploads
    def _registry_path(self) -> str:
        return os.path.join(self.cache_dir, "uploads.json")

    def upload(self, path: str) -> Dict[str, Any]:
        """Upload once per content hash and reuse until shortly before expiry."""
        sha = file_sha256(path)
        registry = read_json(self._registry_path()) if os.path.isfile(self._registry_path()) else {}
        entry = registry.get(sha)
        if entry and time.time() - float(entry["uploaded_unix"]) < UPLOAD_REUSE_SECONDS:
            try:
                remote = self.client.files.get(name=entry["name"])
                if getattr(getattr(remote, "state", None), "name", "") == "ACTIVE":
                    return entry
            except Exception:
                pass
        remote = self.client.files.upload(file=path)
        poll = float(self.cfg.get("upload_poll_seconds", 5.0))
        while True:
            state = getattr(getattr(remote, "state", None), "name", "")
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise GeminiError(f"upload of {path} failed during processing")
            time.sleep(poll)
            remote = self.client.files.get(name=remote.name)
        entry = {"name": remote.name, "uri": remote.uri,
                 "mime_type": getattr(remote, "mime_type", None) or "video/mp4",
                 "sha256": sha, "path": path, "uploaded": utc_now(), "uploaded_unix": time.time()}
        registry[sha] = entry
        write_json(self._registry_path(), registry)
        return entry

    def _clip(self, part: VideoPart) -> str:
        """A local clip file for backends that take no offsets."""
        from .prepare_videos import trim_video

        sha = file_sha256(part.path)[:16]
        out = os.path.join(self.cache_dir, "clips", f"{sha}_{part.start_frame}_{part.end_frame}.mp4")
        if not os.path.isfile(out):
            trim_video(part.path, out, int(part.start_frame), int(part.end_frame), part.source_fps)
        return out

    # ------------------------------------------------------------ calls
    def request_key(self, parts: Sequence[Part], schema: Mapping[str, Any]) -> str:
        described: List[Any] = []
        for part in parts:
            if isinstance(part, TextPart):
                described.append({"text": part.text})
            else:
                described.append({"video": file_sha256(part.path), "fps": float(part.fps),
                                  "start": part.start_frame, "end": part.end_frame})
        payload = canonical_json({"settings": self.settings(), "parts": described, "schema": schema})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _save_failure(self, key: str, label: str, attempt: int, error: str, text: Optional[str], usage: Any,
                      finish: Optional[str]) -> str:
        """One file per failed attempt; the random suffix keeps failures of one request apart however
        close together they happen."""
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        path = os.path.join(self.cache_dir, "failures", f"{key[:16]}_{stamp}_{uuid.uuid4().hex[:12]}.json")
        write_json(path, {"key": key, "label": label, "attempt": attempt, "created": utc_now(), "error": error,
                          "settings": self.settings(), "finish_reason": finish, "usage": usage,
                          "usage_normalized": normalize_usage(usage), "text": text})
        return path

    def generate_json(self, parts: Sequence[Part], schema: Mapping[str, Any], label: str
                      ) -> Tuple[Any, Dict[str, Any]]:
        key = self.request_key(parts, schema)
        path = os.path.join(self.cache_dir, "responses", f"{key}.json")
        if self.use_response_cache and os.path.isfile(path):
            record = read_json(path)
            return record["parsed"], {**{k: v for k, v in record.items() if k != "parsed"}, "cached": True}

        max_errors = int(self.cfg.get("max_retries", 6))
        malformed_left = int(self.cfg.get("malformed_json_retries", 1))
        base = float(self.cfg.get("retry_base_seconds", 5.0))
        errors = 0
        attempts: List[Dict[str, Any]] = []

        def record_attempt(outcome: str, usage: Any = None, finish: Optional[str] = None,
                           saved: Optional[str] = None) -> None:
            attempts.append({"attempt": len(attempts) + 1, "outcome": outcome, "finish_reason": finish,
                             "usage_normalized": normalize_usage(usage), "saved": saved})

        while True:
            started = time.time()
            try:
                if self.backend == "generate_content":
                    text, usage, finish = self._generate_content(parts, schema)
                else:
                    text, usage, finish = self._interactions(parts, schema)
            except Exception as exc:
                errors += 1
                record_attempt(f"error: {type(exc).__name__}")
                if not self._retriable(exc) or errors >= max_errors:
                    raise GeminiRequestFailed(f"{label}: request failed: {exc}", attempts) from exc
                delay = base * (2 ** (errors - 1))
                print(f"[gemini] {label}: {type(exc).__name__}: {exc}; retry {errors}/{max_errors - 1} "
                      f"in {delay:.0f}s", flush=True)
                time.sleep(delay)
                continue
            number = len(attempts) + 1
            if is_truncated(finish):
                saved = self._save_failure(key, label, number, "truncated", text, usage, finish)
                record_attempt("truncated", usage, finish, saved)
                raise GeminiTruncated(f"{label}: the response reached the output limit ({finish}); "
                                      f"raw response saved to {saved}", attempts)
            if not text:
                saved = self._save_failure(key, label, number, "empty", text, usage, finish)
                record_attempt("empty", usage, finish, saved)
                raise GeminiError(f"{label}: empty response (finish reason {finish}); saved to {saved}", attempts)
            try:
                parsed = parse_json_text(text)
            except (json.JSONDecodeError, ValueError) as exc:
                saved = self._save_failure(key, label, number, f"malformed: {exc}", text, usage, finish)
                record_attempt("malformed", usage, finish, saved)
                if malformed_left > 0:
                    malformed_left -= 1
                    print(f"[gemini] {label}: malformed JSON ({exc}); asking once more (saved to {saved})", flush=True)
                    continue
                raise GeminiMalformed(f"{label}: malformed JSON: {exc}; raw response saved to {saved}",
                                      attempts) from exc
            record_attempt("ok", usage, finish)
            record = {
                "key": key, "label": label, "created": utc_now(),
                "seconds": round(time.time() - started, 2), "settings": self.settings(),
                "usage": usage, "usage_normalized": normalize_usage(usage),
                # What the call cost: this answer and every failed attempt before it.
                "usage_total": sum_usage(a["usage_normalized"] for a in attempts),
                "attempts": attempts, "finish_reason": finish,
                "text": text, "parsed": parsed,
            }
            write_json(path, record)
            return parsed, {**{k: v for k, v in record.items() if k != "parsed"}, "cached": False}

    @staticmethod
    def _retriable(exc: BaseException) -> bool:
        if isinstance(exc, (ValueError, TypeError, GeminiError)):
            return False    # SDK validation errors and our own: asking again cannot fix them
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if isinstance(code, int):
            return code in RETRIABLE_CODES
        name = type(exc).__name__.lower()
        return any(word in name for word in ("timeout", "connection", "unavailable", "servererror"))

    def _generate_content(self, parts: Sequence[Part], schema: Mapping[str, Any]):
        from google.genai import types

        content_parts = []
        for part in parts:
            if isinstance(part, TextPart):
                content_parts.append(types.Part(text=part.text))
                continue
            uploaded = self.upload(part.path)
            metadata = {"fps": float(part.fps)}
            if part.start_frame is not None:
                metadata["start_offset"] = f"{part.start_frame / part.source_fps:.3f}s"
            if part.end_frame is not None:
                metadata["end_offset"] = f"{(part.end_frame + 1) / part.source_fps:.3f}s"
            content_parts.append(types.Part(
                file_data=types.FileData(file_uri=uploaded["uri"], mime_type=uploaded["mime_type"]),
                video_metadata=types.VideoMetadata(**metadata),
            ))
        settings = self.settings()
        config: Dict[str, Any] = {
            "max_output_tokens": settings["max_output_tokens"],
            "response_mime_type": "application/json",
        }
        if settings["temperature"] is not None:
            config["temperature"] = settings["temperature"]
        resolution = settings["media_resolution"]
        if resolution != "default":
            config["media_resolution"] = getattr(types.MediaResolution, f"MEDIA_RESOLUTION_{resolution.upper()}")
        try:
            generation = types.GenerateContentConfig(**config, response_json_schema=dict(schema))
        except Exception:
            # SDKs before response_json_schema take the OpenAPI subset instead.
            generation = types.GenerateContentConfig(**config, response_schema=dict(schema))
        contents = [types.Content(role="user", parts=content_parts)]
        if self.cfg.get("preflight_tokens", False):
            self._check_token_budget(contents)
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=generation,
        )
        finish = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish = str(getattr(candidates[0], "finish_reason", None))
        try:
            text = response.text
        except Exception:
            text = None
        return text, _to_dict(getattr(response, "usage_metadata", None)), finish

    def _check_token_budget(self, contents):
        """Check video/text input before generation, reserving headroom for schema and output."""
        if self._model_info is None:
            self._model_info = self.client.models.get(model=self.model)
        limit = self._model_info.input_token_limit
        output_limit = self._model_info.output_token_limit
        requested = self.settings()["max_output_tokens"]
        if output_limit and requested > output_limit:
            raise GeminiError(f"max_output_tokens={requested} exceeds {self.model}'s {output_limit}; lower it")
        count = self.client.models.count_tokens(model=self.model, contents=contents).total_tokens
        if count is None or not limit:
            raise GeminiError("token preflight returned no count or input limit; cannot verify episode fits")
        reserve = requested + int(self.cfg.get("input_token_headroom", 8192))
        print(f"[gemini] input token count {count}; reserved {reserve}; model input limit {limit}", flush=True)
        if count + reserve > limit:
            raise GeminiError(f"whole episode needs {count} input tokens plus {reserve} reserved, exceeding "
                              f"{limit}; lower annotation.videos.fps and prepare videos again, or use a "
                              "model with a larger context. Splitting the answer does not reduce video input.")

    def _interactions(self, parts: Sequence[Part], schema: Mapping[str, Any]):
        settings = self.settings()
        resolution = settings["media_resolution"]
        inputs: List[Dict[str, Any]] = []
        for part in parts:
            if isinstance(part, TextPart):
                inputs.append({"type": "text", "text": part.text})
                continue
            path = part.path
            if part.start_frame is not None or part.end_frame is not None:
                path = self._clip(part)
            uploaded = self.upload(path)
            video = {"type": "video", "uri": uploaded["uri"], "mime_type": uploaded["mime_type"],
                     "processing": {"type": "static", "fps": float(part.fps)}}
            if resolution != "default":
                video["resolution"] = resolution
            inputs.append(video)
        generation_config: Dict[str, Any] = {"max_output_tokens": settings["max_output_tokens"]}
        if settings["temperature"] is not None:
            generation_config["temperature"] = settings["temperature"]
        interaction = self.client.interactions.create(
            model=self.model,
            input=inputs,
            generation_config=generation_config,
            response_format={"type": "text", "mime_type": "application/json", "schema": dict(schema)},
        )
        usage = _to_dict(getattr(interaction, "usage", None))
        status = str(getattr(interaction, "status", None))
        if status.lower().endswith("failed"):
            raise GeminiError(f"interaction failed: {status}")
        return getattr(interaction, "output_text", None), usage, status
