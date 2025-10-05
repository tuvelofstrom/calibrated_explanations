"""Plugin registry implementing ADR-006 trust model."""

from __future__ import annotations

import hashlib
import inspect
import os
import warnings
from copy import deepcopy
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple

from .base import ExplainerPlugin, validate_plugin_meta

TRUST_ENV_VAR = "CE_TRUST_PLUGIN"
_BUILTIN_PROVIDER = "calibrated_explanations"
_ENTRYPOINT_GROUP = "calibrated_explanations.plugins"


@dataclass
class PluginRecord:
    """Internal record storing plugin object and metadata."""

    plugin: ExplainerPlugin
    meta: Dict[str, Any]
    source: str
    trusted: bool
    checksum_status: str


_REGISTRY: MutableMapping[str, PluginRecord] = {}
_WARNED_UNTRUSTED: set[str] = set()


def register(
    plugin: ExplainerPlugin,
    *,
    source: str = "manual",
    built_in: bool | None = None,
) -> PluginRecord:
    """Register a plugin instance and return its :class:`PluginRecord`.

    Metadata is validated according to ADR-006. If the plugin is not trusted it
    remains discoverable via :func:`list_plugins` but will not be returned by
    :func:`find_for` when ``trusted_only`` is ``True``.
    """

    meta = getattr(plugin, "plugin_meta", None)
    validate_plugin_meta(meta)
    meta_copy = deepcopy(meta)
    meta_copy["capabilities"] = tuple(meta_copy["capabilities"])
    name = meta_copy["name"]

    prior_record = _REGISTRY.get(name)

    trusted_names = _trusted_names_from_env()
    provider = meta_copy["provider"]
    is_builtin = bool(built_in) if built_in is not None else provider == _BUILTIN_PROVIDER

    trusted = is_builtin or name in trusted_names
    if prior_record and prior_record.trusted:
        trusted = True

    checksum_status = _verify_checksum(plugin, meta_copy)

    record = PluginRecord(
        plugin=plugin,
        meta=meta_copy,
        source=source,
        trusted=trusted,
        checksum_status=checksum_status,
    )
    _REGISTRY[name] = record

    if not record.trusted:
        _emit_untrusted_warning(record)

    return record


def unregister(identifier: str | ExplainerPlugin) -> None:
    """Remove a plugin from the registry (trusted state included)."""

    name = _resolve_name(identifier)
    _REGISTRY.pop(name, None)
    _WARNED_UNTRUSTED.discard(name)


def clear() -> None:
    """Clear all registered plugins."""

    _REGISTRY.clear()
    _WARNED_UNTRUSTED.clear()


def list_plugins(*, include_untrusted: bool = True) -> Tuple[Mapping[str, Any], ...]:
    """Return plugin metadata records.

    Each record contains the plugin metadata plus derived fields:

    - ``trusted``: whether the plugin is currently trusted
    - ``source``: registration source (manual, entry point, etc.)
    - ``checksum_status``: ``ok``, ``mismatch``, ``unverified``, or ``not-provided``
    """

    records: Iterable[PluginRecord]
    if include_untrusted:
        records = _REGISTRY.values()
    else:
        records = (record for record in _REGISTRY.values() if record.trusted)
    return tuple(_public_record(record) for record in records)


def trust_plugin(identifier: str | ExplainerPlugin) -> None:
    """Mark a plugin as trusted by name or plugin object."""

    name = _resolve_name(identifier)
    record = _REGISTRY.get(name)
    if record is None:
        raise ValueError(f"Plugin '{name}' is not registered")
    record.trusted = True


def untrust_plugin(identifier: str | ExplainerPlugin) -> None:
    """Mark a plugin as untrusted if present."""

    name = _resolve_name(identifier)
    record = _REGISTRY.get(name)
    if record is not None:
        record.trusted = False


def find_for(model: Any, *, trusted_only: bool = False) -> Tuple[ExplainerPlugin, ...]:
    """Return plugins that report support for ``model``.

    ``trusted_only`` defaults to ``False`` to allow manual experimentation, but
    callers that rely on ADR-006 guarantees should pass ``trusted_only=True`` or
    use :func:`find_for_trusted`.
    """

    records: Iterable[PluginRecord]
    if trusted_only:
        records = (record for record in _REGISTRY.values() if record.trusted)
    else:
        records = _REGISTRY.values()
    return tuple(record.plugin for record in records if _safe_supports(record.plugin, model))


def find_for_trusted(model: Any) -> Tuple[ExplainerPlugin, ...]:
    """Return trusted plugins that support ``model``."""

    return find_for(model, trusted_only=True)


def discover_entrypoint_plugins(group: str = _ENTRYPOINT_GROUP) -> Tuple[str, ...]:
    """Load plugins exposed via the setuptools entry point group.

    Returns a tuple of plugin names that were discovered. Untrusted plugins are
    still registered (so that metadata is visible) but remain untrusted until the
    user opts in via :func:`trust_plugin` or :envvar:`CE_TRUST_PLUGIN`.
    """

    try:
        entry_points = importlib_metadata.entry_points()
    except Exception as exc:  # pragma: no cover - importlib metadata failure is rare
        warnings.warn(f"Failed to load plugin entry points: {exc}")
        return ()

    if hasattr(entry_points, "select"):
        selected = entry_points.select(group=group)
    else:  # pragma: no cover - Python <3.10 compatibility
        selected = [ep for ep in entry_points if ep.group == group]

    discovered: list[str] = []
    for entry_point in selected:
        try:
            plugin = entry_point.load()
        except Exception as exc:  # pragma: no cover - plugin import failures
            warnings.warn(f"Failed to load plugin '{entry_point.name}': {exc}")
            continue
        record = register(plugin, source=f"entry_point:{entry_point.name}")
        discovered.append(record.meta["name"])
    return tuple(discovered)


def _public_record(record: PluginRecord) -> Dict[str, Any]:
    result = dict(record.meta)
    result.update(
        {
            "trusted": record.trusted,
            "source": record.source,
            "checksum_status": record.checksum_status,
        }
    )
    return result


def _resolve_name(identifier: str | ExplainerPlugin) -> str:
    if isinstance(identifier, str):
        return identifier
    meta = getattr(identifier, "plugin_meta", None)
    if isinstance(meta, dict) and "name" in meta:
        return str(meta["name"])
    raise ValueError("Cannot resolve plugin name from identifier")


def _trusted_names_from_env() -> set[str]:
    raw = os.getenv(TRUST_ENV_VAR, "")
    if not raw:
        return set()
    separators = {",", os.pathsep}
    for sep in list(separators):
        raw = raw.replace(sep, ",")
    names = {chunk.strip() for chunk in raw.split(",") if chunk.strip()}
    return names


def _emit_untrusted_warning(record: PluginRecord) -> None:
    name = record.meta["name"]
    if name in _WARNED_UNTRUSTED:
        return
    _WARNED_UNTRUSTED.add(name)
    warnings.warn(
        (
            f"Plugin '{name}' from provider '{record.meta['provider']}' is not trusted. "
            f"Set {TRUST_ENV_VAR}={name} or call trust_plugin('{name}') to opt in."
        ),
        UserWarning,
        stacklevel=3,
    )


def _verify_checksum(plugin: ExplainerPlugin, meta: Mapping[str, Any]) -> str:
    expected = meta.get("checksum_sha256")
    if not expected:
        return "not-provided"

    module = inspect.getmodule(plugin.__class__)
    path = getattr(module, "__file__", None) if module else None
    if not path:
        return "unverified"
    try:
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return "unverified"
    return "ok" if digest == expected.lower() else "mismatch"


def _safe_supports(plugin: ExplainerPlugin, model: Any) -> bool:
    try:
        return bool(plugin.supports(model))
    except Exception:
        return False


__all__ = [
    "TRUST_ENV_VAR",
    "register",
    "unregister",
    "clear",
    "list_plugins",
    "trust_plugin",
    "untrust_plugin",
    "find_for",
    "find_for_trusted",
    "discover_entrypoint_plugins",
]

