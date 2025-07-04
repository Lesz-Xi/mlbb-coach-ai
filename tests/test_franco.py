import sys
import os
import pytest

# Add parent directory to path for imports to allow finding the 'rules' module
# This is a common pattern for local package development.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rules.roles.tank.franco import evaluate


@pytest.fixture
def franco_data():
    """Base data for a 15-minute Franco match."""
    return {
        "hero": "franco",
        "kills": 2, "deaths": 5, "assists": 10,  # KDA: 2.4
        "gold_per_min": 450,
        "hooks_landed": 5,
        "team_engages": 4,
        "vision_score": 6,
        "hero_damage": 20000,
        "turret_damage": 500,
        "damage_taken": 65000,  # 4333/min
        "teamfight_participation": 60,
        "positioning_rating": "good",
        "ult_usage": "average",
        "match_duration": 15
    }


def test_kda_tank_thresholds(franco_data):
    """Test that tank KDA thresholds are used (2.5-4.0)."""
    # KDA is 2.4, which is just below the 2.5 low threshold.
    franco_data["deaths"] = 4  # KDA becomes 3.0
    out = evaluate(franco_data)
    # With a KDA of 3.0, should be neutral/info feedback
    assert not any("warning:" in msg for msg in out)
    assert any("Decent KDA" in msg for msg in out)

    franco_data["deaths"] = 6  # KDA becomes 2.0
    out = evaluate(franco_data)
    assert any("KDA 2.0 (< 2.5)" in msg for msg in out)


def test_damage_taken_scaling(franco_data):
    """Test damage taken scaling (4000/min)."""
    # At 15 mins, needs 60,000. Current is 65,000 (good).
    out = evaluate(franco_data)
    assert not any("Damage taken" in msg for msg in out)

    # Low damage taken should trigger a warning.
    franco_data["damage_taken"] = 50000  # Below 60,000 threshold
    out = evaluate(franco_data)
    assert any("Damage taken 50,000" in msg for msg in out)


def test_franco_specific_metrics(franco_data):
    """Test hero-specific checks for hooks, engages, and vision."""
    # Low hooks landed
    franco_data["hooks_landed"] = 2
    out = evaluate(franco_data)
    assert any("Only 2 hooks landed (< 4)" in msg for msg in out)

    # Low team engages
    franco_data["team_engages"] = 1
    out = evaluate(franco_data)
    assert any("Only 1 team engages (< 3)" in msg for msg in out)

    # Low vision score
    franco_data["vision_score"] = 2
    out = evaluate(franco_data)
    assert any("Vision score 2 (< 5)" in msg for msg in out)


def test_high_deaths(franco_data):
    """Test critical feedback for excessive deaths."""
    franco_data["deaths"] = 10
    out = evaluate(franco_data)
    assert any("critical: 10 deaths is too high" in msg for msg in out)


def test_no_franco_specific_keys(franco_data):
    """Test that missing hero-specific keys doesn't crash."""
    del franco_data["hooks_landed"]
    del franco_data["team_engages"]
    # Should still run and provide feedback on other metrics
    out = evaluate(franco_data)
    assert isinstance(out, list)
    assert len(out) > 0

    # It should provide warnings about the missing metrics
    assert any("hooks landed" in msg for msg in out)
    assert any("team engages" in msg for msg in out) 