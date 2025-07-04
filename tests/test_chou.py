# Unit tests for Chou evaluator
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rules.roles.fighter.chou import evaluate


def test_kda_low():
    """Test low KDA detection."""
    data = {"kills": 2, "assists": 1, "deaths": 2}
    out = evaluate(data, minutes=15)
    assert any("KDA 1.5" in msg for msg in out)
    print("✓ test_kda_low passed")


def test_kda_high():
    """Test high KDA praise."""
    data = {"kills": 8, "assists": 4, "deaths": 2}
    out = evaluate(data, minutes=15)
    assert any("Great KDA 6.0" in msg for msg in out)
    print("✓ test_kda_high passed")


def test_no_participation_key():
    """Test missing teamfight participation."""
    data = {}
    out = evaluate(data, minutes=12)
    assert any("Low fight presence" in msg for msg in out)
    print("✓ test_no_participation_key passed")


def test_damage_scaling():
    """Test damage scaling with match duration."""
    # 10 minute game - expect 37,500 damage minimum
    data = {"hero_damage": 30000}
    out = evaluate(data, minutes=10)
    assert any("Damage 30,000" in msg for msg in out)
    
    # 20 minute game - expect 75,000 damage minimum
    data = {"hero_damage": 60000}
    out = evaluate(data, minutes=20)
    assert any("Damage 60,000" in msg for msg in out)
    print("✓ test_damage_scaling passed")


def test_gpm_scaling():
    """Test GPM scaling with match duration."""
    # Early game (< 10 min) - lower threshold
    data = {"gold_per_min": 500}
    out = evaluate(data, minutes=8)
    # 650 * 0.8 = 520, so 500 should trigger warning
    assert any("GPM 500" in msg for msg in out)
    
    # Late game (> 20 min) - higher threshold  
    data = {"gold_per_min": 700}
    out = evaluate(data, minutes=25)
    # 650 * 1.1 = 715, so 700 should trigger warning
    assert any("GPM 700" in msg for msg in out)
    print("✓ test_gpm_scaling passed")


def test_chou_specific():
    """Test Chou-specific checks."""
    # High deaths
    data = {"deaths": 8}
    out = evaluate(data, minutes=15)
    assert any("8 deaths is too high" in msg for msg in out)
    
    # Low damage ratio
    data = {"hero_damage": 20000, "damage_taken": 25000}
    out = evaluate(data, minutes=15)
    assert any("Damage ratio 0.8" in msg for msg in out)
    
    # Low ult usage
    data = {"ult_usage": "low"}
    out = evaluate(data, minutes=15)
    assert any("Way of Dragon" in msg for msg in out)
    print("✓ test_chou_specific passed")


def test_severity_levels():
    """Test that severity levels are included in output."""
    data = {"deaths": 10}  # Should trigger critical
    out = evaluate(data, minutes=15)
    assert any("critical:" in msg for msg in out)
    
    data = {"gold_per_min": 400}  # Should trigger warning
    out = evaluate(data, minutes=15)
    assert any("warning:" in msg for msg in out)
    print("✓ test_severity_levels passed")


if __name__ == "__main__":
    print("Running Chou evaluator tests...\n")
    
    test_kda_low()
    test_kda_high()
    test_no_participation_key()
    test_damage_scaling()
    test_gpm_scaling()
    test_chou_specific()
    test_severity_levels()
    
    print("\n✅ All tests passed!")
    print("\nPinning these now prevents silent regressions when you "
          "tweak constants next month.") 