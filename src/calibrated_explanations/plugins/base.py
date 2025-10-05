"""Plugin base interfaces (ADR-006 skeleton).

Minimal interfaces to support a registry of third-party explainers. This is an
opt-in surface; users should understand that loading external plugins executes
arbitrary code. We will document risks and keep the registry explicit.

Contract (v0.1, unstable):
- Each plugin module exposes a ``plugin_meta`` dict with at least:
    {"schema_version": 1, "capabilities": ["explain"], "name": str}
- Each plugin exposes two callables:
    supports(model) -> bool
    explain(model, X, **kwargs) -> Any  # typically an Explanation or legacy dict

This mirrors ADR-006 minimal capability metadata and keeps behavior opt-in.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Protocol


class ExplainerPlugin(Protocol):
    """Protocol for explainer plugins.

    Implementations are expected to provide:
    - plugin_meta: Dict[str, Any]
    - supports(model) -> bool
    - explain(model, X, **kwargs) -> Any
    """

    plugin_meta: Dict[str, Any]

    def supports(self, model: Any) -> bool:  # pragma: no cover - protocol
        ...

    def explain(self, model: Any, X: Any, **kwargs: Any) -> Any:  # pragma: no cover - protocol
        ...


def validate_plugin_meta(meta: Dict[str, Any]) -> None:
    """Validate minimal plugin metadata.

    Required keys (ADR-006):
    - ``schema_version`` (int)
    - ``name`` (str, non-empty)
    - ``version`` (str, non-empty)
    - ``provider`` (str, non-empty)
    - ``capabilities`` (iterable of ``str``)

    Optional keys:
    - ``checksum_sha256`` (64-character hexadecimal string)
    """

    if not isinstance(meta, dict):
        raise ValueError("plugin_meta must be a dict")

    required_types = {
        "schema_version": int,
        "name": str,
        "version": str,
        "provider": str,
    }
    for key, typ in required_types.items():
        if key not in meta:
            raise ValueError(f"plugin_meta missing required key: {key}")
        if not isinstance(meta[key], typ):
            raise ValueError(f"plugin_meta[{key!r}] must be {typ.__name__}")
        if isinstance(meta[key], str) and not meta[key].strip():
            raise ValueError(f"plugin_meta[{key!r}] must be a non-empty string")

    if "capabilities" not in meta:
        raise ValueError("plugin_meta missing required key: capabilities")

    capabilities = meta["capabilities"]
    if not isinstance(capabilities, Iterable) or isinstance(capabilities, (str, bytes)):
        raise ValueError("plugin_meta['capabilities'] must be an iterable of strings")
    if not all(isinstance(cap, str) and cap.strip() for cap in capabilities):
        raise ValueError("plugin_meta['capabilities'] entries must be non-empty strings")

    checksum = meta.get("checksum_sha256")
    if checksum is not None:
        if not isinstance(checksum, str):
            raise ValueError("plugin_meta['checksum_sha256'] must be a string if provided")
        hex_value = checksum.strip().lower()
        if len(hex_value) != 64 or any(ch not in "0123456789abcdef" for ch in hex_value):
            raise ValueError("plugin_meta['checksum_sha256'] must be a 64-character hex digest")


__all__ = ["ExplainerPlugin", "validate_plugin_meta"]
