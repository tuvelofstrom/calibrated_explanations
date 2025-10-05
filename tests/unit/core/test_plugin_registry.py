from typing import Any, Dict

import pytest

from calibrated_explanations.plugins import registry


class DummyPlugin:
    plugin_meta: Dict[str, Any] = {
        "schema_version": 1,
        "capabilities": ["explain"],
        "name": "dummy-alt",
        "version": "0.0.1",
        "provider": "tests",
    }

    def supports(self, model: Any) -> bool:
        return getattr(model, "is_dummy", False)

    def explain(self, model: Any, X: Any, **kwargs: Any) -> Any:
        return {"ok": True}


def test_register_and_find_for():
    registry.clear()
    p = DummyPlugin()
    registry.register(p)

    class M:
        is_dummy = True

    assert registry.find_for_trusted(M()) == ()

    registry.trust_plugin(p)
    found = registry.find_for_trusted(M())
    assert found == (p,)


def test_register_validation():
    registry.clear()

    class Bad:
        plugin_meta = {"name": "bad", "capabilities": ["explain"]}

        def supports(self, model: Any) -> bool:
            return False

        def explain(self, model: Any, X: Any, **kwargs: Any) -> Any:
            return None

    with pytest.raises(ValueError):
        registry.register(Bad())


def test_unregister():
    registry.clear()

    class P(DummyPlugin):
        plugin_meta = DummyPlugin.plugin_meta | {"name": "dummy-unregister"}

    p = P()
    registry.register(p)
    assert registry.list_plugins()[0]["name"] == p.plugin_meta["name"]
    registry.unregister(p)
    assert registry.list_plugins() == ()
