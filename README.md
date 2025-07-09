# SkillShift AI

**Intelligent Coaching System for Mobile Legends: Bang Bang Players**

---

## 🎯 Project Goal

SkillShift AI is an advanced AI-powered coaching tool that analyzes Mobile Legends: Bang Bang match statistics and provides actionable, role-specific feedback. Built with sophisticated evaluation logic and dynamic thresholds, it simulates what a professional coach would tell you—based on real performance data and game knowledge.

## ⚡ Key Features

- **🎮 Role-Specific Analysis** - Tailored evaluation for all 6 MLBB roles (Marksman, Assassin, Mage, Tank, Support, Fighter)
- **🧠 Mental Boost Feedback** - Contextual encouragement based on performance trends
- **⏱️ Dynamic Scaling** - Performance thresholds adjust based on match duration
- **🎯 Severity Levels** - Prioritized feedback (Critical → Warning → Info → Success)
- **📊 Comprehensive Metrics** - KDA, GPM, damage ratios, positioning, teamfight participation
- **⚙️ Configurable Thresholds** - YAML-based configuration for easy tuning
- **🧪 Battle-Tested** - Comprehensive unit testing and validation

## 🚀 What Makes It Advanced

### Smart Evaluation Engine

- **Match Duration Scaling**: 10-minute games have different expectations than 25-minute games
- **Context-Aware Feedback**: "Your 65,000 damage in 17 minutes is below the 63,750 threshold"
- **Neutral Feedback**: Always provides context, even for average performance

### Sophisticated Architecture

- **BaseEvaluator Pattern**: Shared logic with hero-specific overrides
- **Configuration-Driven**: No hardcoded thresholds—everything in YAML
- **Type Safety**: Full type hints for reliability

### Mental Boost Feedback

The system now includes a `MentalCoach` that analyzes your performance trend over recent matches. It determines if you're on an upward, downward, or consistent trajectory and provides tailored psychological encouragement. This goes beyond stats to support the mental aspect of competitive gaming.

## 📁 Current Architecture

```
skillshift-ai/
├── main.py                    # Main script to run the coach
├── coach.py                   # Handles dynamic rule loading and evaluation
├── requirements.txt           # Project dependencies
├── core/
│   ├── base_evaluator.py      # Core class with shared evaluation logic
│   ├── data_collector.py      # Handles loading and validating match data
│   ├── mental_coach.py        # Provides encouragement based on history
│   └── schemas.py             # Pydantic schemas for data validation
├── config/
│   └── thresholds.yml         # Role/hero-specific performance thresholds
├── data/
│   ├── player_history.json    # Simulated player performance history
│   └── sample_match.json      # Test data for multiple heroes
├── rules/roles/               # Hero-specific evaluation logic
│   ├── assassin/lancelot.py
│   ├── fighter/chou.py
│   ├── mage/kagura.py
│   ├── marksman/miya.py
│   ├── support/estes.py
│   └── tank/
│       ├── franco.py
│       └── tigreal.py
├── tests/
│   ├── test_chou.py           # Unit tests for Chou
│   └── test_franco.py         # Unit tests for Franco
└── docs/
    └── refactoring_summary.md # Architecture and refactoring details
```

## 🤖 Tech Stack

**Current Implementation:**

- **Python 3.x** with type hints
- **Pydantic** for data validation
- **YAML** for configuration management
- **Dynamic imports** for modular hero evaluation
- **Inheritance patterns** for code reuse
- **Unit testing** for reliability

**Planned Enhancements:**

- **Web Interface** (React + Tailwind + GSAP)
- **ML Integration** (scikit-learn, pandas)
- **API Integration** (MLBB official data sources)

## 📊 Example Output

```bash
🧠 Match 1 Coaching Report (Hero: Chou)
- warning: KDA 2.4 (< 3.0). Pick safer angles and exit with 2nd skill.
- warning: GPM 580 (< 650). Clear side waves between ganks.
- warning: Low fight presence (58% < 60%). Collapse faster on ally engages.
- critical: 7 deaths is too high. Chou has great mobility—use Shunpo to disengage after combos.

멘탈 코칭:
- Your KDA is trending up (3.8 -> 4.5). Keep up the great work! Your map awareness is clearly improving.
```

## ⚙️ Configuration Example

```yaml
# config/thresholds.yml
heroes:
  chou:
    damage_base: 3750 # Slightly lower than generic fighter
    teamfight_participation: 60

roles:
  fighter:
    kda:
      low: 3.0
      high: 5.0
    gpm_base: 650
```

## 🧪 Testing & Quality

```bash
# Install dependencies
pip install -r requirements.txt

# Run comprehensive tests
python -m unittest discover tests

# Example for a single test file
python tests/test_franco.py
```

## 📌 Current Status

- ✅ **Production-Ready Core** - Advanced evaluation engine with dynamic thresholds
- ✅ **7 Hero Implementations** - Coverage of major MLBB roles and heroes
- ✅ **Contextual Encouragement** - Mental coaching based on performance trends
- ✅ **Comprehensive Testing** - Unit tests prevent regressions
- ✅ **Professional Architecture** - Inheritance, configuration, modularity
- 🔜 **Next Phase**: Web UI and ML integration

## 🧠 About the Developer

Built by **Rhine (Les)** - a 400+ star Mythical-ranked MLBB player combining deep game knowledge with software engineering expertise. This project demonstrates advanced Python architecture while solving real gameplay improvement challenges.

## 💡 Why This Matters

Traditional MLBB tools provide raw statistics without context. This system provides **intelligent coaching**:

- **Context-Aware**: "Your 580 GPM in a 17-minute game needs improvement"
- **Actionable**: "Use Shunpo to disengage after combos"
- **Prioritized**: Critical issues highlighted first
- **Role-Specific**: Marksman advice differs from Tank strategies

---

## 🚀 Quick Start

```bash
# Clone and run
git clone [your-repo]
cd skillshift-ai
pip install -r requirements.txt
python main.py

# Run tests
python -m unittest discover tests
```

_Ready for production use with plans for web deployment and ML enhancement._
