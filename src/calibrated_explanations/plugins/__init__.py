"""Explainer plugin interfaces and registry (ADR-006, minimal skeleton).

This subpackage exposes:
- ``ExplainerPlugin`` protocol and ``validate_plugin_meta`` helper
- ``registry`` utilities: register, unregister, list_plugins, find_for

Security note: Registering/using third-party plugins executes arbitrary code.
Only use plugins you trust. This API is opt-in and intentionally explicit.
"""

from . import registry  # re-export module for convenience
from .base import ExplainerPlugin, validate_plugin_meta  # noqa: F401
from .registry import (  # noqa: F401
    TRUST_ENV_VAR,
    discover_entrypoint_plugins,
    find_for_trusted,
    trust_plugin,
    untrust_plugin,
)

__all__ = [
    "ExplainerPlugin",
    "validate_plugin_meta",
    "registry",
    "TRUST_ENV_VAR",
    "trust_plugin",
    "untrust_plugin",
    "find_for_trusted",
    "discover_entrypoint_plugins",
]
