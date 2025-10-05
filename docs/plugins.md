# Plugin registry and ADR-006 trust workflow

ADR-006 introduced a conservative trust model for third-party plugins. The
registry shipped in this repository now implements that model end-to-end:

- Plugins are discovered either explicitly (manual registration) or via the
  ``calibrated_explanations.plugins`` entry-point group.
- Metadata is validated for required fields (name, version, provider,
  capabilities, optional checksum) before registration succeeds.
- Only trusted plugins participate in automated discovery. Trust can be granted
  via :envvar:`CE_TRUST_PLUGIN` or programmatically with
  :func:`calibrated_explanations.plugins.trust_plugin`.
- Warnings are emitted the first time an untrusted plugin is seen so that users
  understand how to opt-in.
- Diagnostic helpers expose registered metadata via
  :func:`calibrated_explanations.plugins.registry.list_plugins` (with the option
  to include or filter untrusted entries).
- Optional SHA256 checksums supplied by plugin authors are verified on a
  best-effort basis to detect tampering.

The sections below show how to work with the registry safely.

## Quick start

```py
from calibrated_explanations.plugins import registry, TRUST_ENV_VAR
from tests.plugins.example_plugin import PLUGIN

# Start from a clean slate (primarily useful in tests)
registry.clear()

# Registering validates metadata and records diagnostics. Untrusted plugins
# trigger a one-time warning and are excluded from trusted discovery helpers.
record = registry.register(PLUGIN)
print(record.trusted)  # False by default for third-party plugins

# ``list_plugins`` returns immutable metadata dictionaries that include the
# trusted flag, registration source, and checksum verification status.
print(registry.list_plugins())

# Trusted discovery will skip untrusted plugins until you opt-in explicitly.
assert registry.find_for_trusted("supported-model") == ()

# Trust can be granted at runtime...
registry.trust_plugin(PLUGIN)
assert registry.find_for_trusted("supported-model") == (PLUGIN,)

# ...or via environment variable to support repeatable deployments.
# export CE_TRUST_PLUGIN="tests.example_plugin"

# When finished, remove or untrust plugins as needed.
registry.untrust_plugin(PLUGIN)
registry.unregister(PLUGIN)
```

> **Tip:** ``CE_TRUST_PLUGIN`` accepts a comma- or path-separated list of plugin
> names. Trust granted via the environment persists across registrations and
> suppresses untrusted warnings.

## Entry-point discovery

Package authors can expose plugins via the ``calibrated_explanations.plugins``
setuptools entry-point group. Consumers can load these entry points with:

```py
from calibrated_explanations.plugins import discover_entrypoint_plugins

discovered = discover_entrypoint_plugins()
print("Registered plugins:", discovered)
```

Plugins discovered this way obey the same trust rules—metadata is recorded, but
untrusted plugins are not returned by trusted discovery helpers until the user
opts in.

## Metadata schema

Plugins must define a ``plugin_meta`` mapping with the following required keys:

- ``schema_version`` (int)
- ``name`` (str)
- ``version`` (str)
- ``provider`` (str)
- ``capabilities`` (iterable of str)

Optional key:

- ``checksum_sha256`` (str): 64-character hexadecimal digest of the plugin
  module file. When provided, the registry recomputes the checksum to detect
  mismatches and reports the status via ``checksum_status`` in the diagnostic
  metadata.

If metadata is malformed a :class:`ValueError` is raised during registration.

## Security considerations

- **Trust is explicit.** Loading or registering a plugin executes arbitrary
  Python code. Trust flags do not sandbox execution; they simply gate automated
  discovery helpers and remind users to opt in intentionally.
- **Warnings are actionable.** The first encounter with an untrusted plugin
  emits a warning including the environment variable and API call required to
  trust it. Subsequent registrations of the same plugin do not spam additional
  warnings.
- **Checksum verification is best-effort.** If a checksum cannot be computed
  (for example, when a module has no ``__file__`` attribute) the registry marks
  the status as ``"unverified"`` but still records the plugin metadata.
- **Built-in plugins auto-trust.** Plugins whose provider is
  ``"calibrated_explanations"`` (or that are explicitly registered with
  ``built_in=True``) are trusted automatically so that official plugins work out
  of the box.

For the rationale behind these decisions, see
``improvement_docs/adrs/ADR-006-plugin-registry-trust-model.md``.
