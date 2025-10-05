import hashlib
import importlib
import types

import pytest

from calibrated_explanations.plugins import registry
from calibrated_explanations.plugins.registry import TRUST_ENV_VAR


def test_register_requires_trust():
    mod = importlib.import_module("tests.plugins.example_plugin")
    plugin = getattr(mod, "PLUGIN")

    registry.clear()

    with pytest.warns(UserWarning) as warn_record:
        record = registry.register(plugin, source="unit-test")

    assert not record.trusted
    assert warn_record[0].message.args[0].startswith("Plugin 'tests.example_plugin'")

    info = registry.list_plugins()
    assert len(info) == 1
    entry = info[0]
    assert entry["name"] == plugin.plugin_meta["name"]
    assert entry["trusted"] is False
    assert entry["source"] == "unit-test"

    assert registry.list_plugins(include_untrusted=False) == ()
    assert registry.find_for_trusted("supported-model") == ()

    # Manual trust enables discovery
    registry.trust_plugin(plugin)
    trusted = registry.find_for_trusted("supported-model")
    assert trusted and trusted[0] is plugin

    # find_for with trusted_only defaults to False to support manual usage
    all_found = registry.find_for("supported-model")
    assert plugin in all_found

    # Untrust removes from trusted discovery but keeps metadata visible
    registry.untrust_plugin(plugin)
    assert registry.find_for_trusted("supported-model") == ()
    assert registry.list_plugins()[0]["trusted"] is False

    registry.unregister(plugin)
    assert registry.list_plugins() == ()


def test_validate_plugin_meta_rejects_bad_meta():
    class BadPlugin:
        plugin_meta = {"capabilities": ["explain"], "name": "bad"}

        def supports(self, model):
            return False

        def explain(self, model, X, **kwargs):
            return {}

    with pytest.raises(ValueError):
        registry.register(BadPlugin())


class DummyPlugin:
    plugin_meta = {
        "schema_version": 1,
        "capabilities": ["explain"],
        "name": "dummy",
        "version": "0.1.0",
        "provider": "tests",
    }

    def supports(self, model):
        return getattr(model, "is_dummy", False)

    def explain(self, model, X, **kwargs):
        return {"explained": True}


def test_register_and_trust_flow(monkeypatch):
    registry.clear()
    p = DummyPlugin()

    registry.register(p)

    with pytest.raises(ValueError):
        registry.trust_plugin("unknown")

    registry.trust_plugin(p)
    trusted = registry.find_for_trusted(types.SimpleNamespace(is_dummy=True))
    assert trusted == (p,)

    registry.untrust_plugin(p)
    assert registry.find_for_trusted(types.SimpleNamespace(is_dummy=True)) == ()

    registry.unregister(p)


def test_env_trust_enables_auto_trust(monkeypatch):
    registry.clear()
    p = DummyPlugin()
    monkeypatch.setenv(TRUST_ENV_VAR, p.plugin_meta["name"])

    record = registry.register(p, source="env-test")
    assert record.trusted is True
    assert registry.list_plugins(include_untrusted=False)[0]["trusted"] is True


def test_checksum_status(monkeypatch, tmp_path):
    registry.clear()

    class ChecksumPlugin(DummyPlugin):
        plugin_meta = DummyPlugin.plugin_meta | {"name": "checksum", "version": "0.2.0"}

    plugin = ChecksumPlugin()
    module_path = __file__

    with open(module_path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()

    plugin.plugin_meta["checksum_sha256"] = digest
    record = registry.register(plugin)
    assert record.checksum_status == "ok"

    plugin.plugin_meta["checksum_sha256"] = "0" * 64
    record2 = registry.register(plugin)
    assert record2.checksum_status == "mismatch"
