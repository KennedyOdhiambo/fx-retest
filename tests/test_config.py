"""Retest config smoke test: config loads and every strategy.md tunable is present."""

from retest import get_config


def test_config_loads_with_expected_sections() -> None:
    cfg = get_config()
    for section in ("data", "levels", "retest", "backtest", "validation", "report"):
        assert cfg.section(section), f"missing section: {section}"


def test_key_tunables_present() -> None:
    cfg = get_config()
    assert cfg.get("levels.swing_lookback") == 4
    assert float(cfg.get("levels.touch_tolerance_atr")) >= 0.1
    assert cfg.get("retest.max_break_return_bars") == 24
    assert cfg.get("retest.touch_gates.3") == 0
    assert float(cfg.get("retest.min_reward_risk")) >= 1.0
    assert cfg.get("missing.key", "fallback") == "fallback"
