"""
PROJECT-TURBO | Smoke tests — bridge loading + basic functionality
No model required. Verifies the adapter layer is functional.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_bridge_loads():
    """Verify turbo_bridge.so loads."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    print("[PASS] Bridge loads OK")


def test_n_vocab():
    """Verify vocab size is accessible after init."""
    from turbo import TurboBridge

    # This requires a model, skip if not available
    model = os.environ.get("TURBO_TEST_MODEL", "")
    if not model or not os.path.exists(model):
        print("[SKIP] No model for vocab test")
        return
    bridge = TurboBridge()
    ret = bridge.init(model, 512, 8, 41, -1, 0, 1)
    assert ret == 0, f"init failed: {ret}"
    nv = bridge.n_vocab()
    assert nv > 0, f"vocab size is 0"
    print(f"[PASS] n_vocab={nv}")
    bridge.free()


def test_turbo_context():
    """Verify TurboContext creation."""
    from turbo import TurboContext

    model = os.environ.get("TURBO_TEST_MODEL", "")
    if not model or not os.path.exists(model):
        print("[SKIP] No model for context test")
        return
    ctx = TurboContext(
        model_path=model,
        n_ctx=512,
        cache_type="turbo3",
        flash_attn=False,
    )
    assert ctx.n_ctx == 512
    assert ctx._n_vocab > 0
    print(f"[PASS] {ctx}")
    ctx.bridge.free()


def test_turbo_type_map():
    """Verify TURBO_TYPE_MAP has expected entries."""
    from turbo import TURBO_TYPE_MAP

    assert "turbo3" in TURBO_TYPE_MAP
    assert "turbo4" in TURBO_TYPE_MAP
    assert "turbo2" in TURBO_TYPE_MAP
    assert TURBO_TYPE_MAP["turbo3"] == 41
    assert TURBO_TYPE_MAP["turbo4"] == 42
    assert TURBO_TYPE_MAP["turbo2"] == 43
    print("[PASS] TURBO_TYPE_MAP correct")


def test_setup():
    """Verify setup() loads bridge."""
    from turbo import setup

    bridge = setup(verbose=False)
    assert bridge is not None
    print("[PASS] setup() OK")


if __name__ == "__main__":
    print("Running PROJECT-TURBO smoke tests...\n")
    test_bridge_loads()
    test_turbo_type_map()
    test_setup()
    test_n_vocab()
    test_turbo_context()
    print("\n[TURBO] All smoke tests completed.")
