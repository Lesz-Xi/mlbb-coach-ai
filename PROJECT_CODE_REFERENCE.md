# MLBB Coach AI - Project Code Reference

## 🎯 Overview

This document provides a comprehensive reference of all vital code files and their functionality in the MLBB Coach AI project. The project is a full-stack AI-powered coaching system for Mobile Legends: Bang Bang gameplay analysis.

---

## 📁 Project Structure

```
skillshift-ai/
├── 🔧 Backend (FastAPI)
├── 🎨 Frontend (Next.js)
├── 🧠 Core AI/ML Engine
├── 📊 Data & Configuration
├── 🎮 Game Rules & Logic
└── 🧪 Testing & Validation
```

---

## 🔧 **BACKEND - FastAPI Application**

### **Primary API Server**

| File         | Functionality                | Lines | Key Features                                                                                  |
| ------------ | ---------------------------- | ----- | --------------------------------------------------------------------------------------------- |
| `web/app.py` | **Main FastAPI Application** | 1409  | Core API endpoints, Ultimate Parsing System integration, Authentication, CORS, Error handling |

### **Specialized API Modules**

| File                           | Functionality                 | Lines | Key Features                                                               |
| ------------------------------ | ----------------------------- | ----- | -------------------------------------------------------------------------- |
| `web/video_analysis_api.py`    | **Video Analysis Endpoint**   | 415   | Temporal analysis, Behavioral modeling, YOLO integration, Frame processing |
| `web/tactical_coaching_api.py` | **Tactical Coaching API**     | 405   | Strategic analysis, Post-game insights, Coaching recommendations           |
| `web/validation_api.py`        | **Validation & Testing API**  | 571   | User feedback collection, Ground truth annotation, Edge case testing       |
| `web/debug_panel.py`           | **Developer Debug Interface** | 817   | Real-time diagnostics, OCR visualization, Analysis debugging               |
| `web/debug_ultimate.py`        | **Ultimate System Debug**     | 260   | High-confidence analysis testing, Performance monitoring                   |

### **Key API Endpoints**

- **`/api/analyze`** - Main screenshot analysis (legacy, heavy processing)
- **`/api/analyze-fast`** - Fast analysis with reduced processing
- **`/api/analyze-instant`** - Lightning-fast analysis (2-second response)
- **`/api/health`** - System health check
- **`/api/health-isolated`** - Isolated health check (never blocks)

---

## 🎨 **FRONTEND - Next.js Application**

### **Core Application Structure**

| File                           | Functionality             | Lines | Key Features                                    |
| ------------------------------ | ------------------------- | ----- | ----------------------------------------------- |
| `dashboard-ui/app/layout.jsx`  | **Root Layout Component** | 24    | Global styling, Metadata, Favicon configuration |
| `dashboard-ui/app/page.jsx`    | **Main Dashboard**        | 188   | Navigation, System overview, Component routing  |
| `dashboard-ui/app/globals.css` | **Global Styles**         | 95    | Tailwind CSS, Custom styling, Dark theme        |

### **Page Components**

| File                                            | Functionality                    | Key Features                                                     |
| ----------------------------------------------- | -------------------------------- | ---------------------------------------------------------------- |
| `dashboard-ui/app/screenshot-analysis/page.jsx` | **Screenshot Upload & Analysis** | File upload, Real-time analysis, Results display, IGN management |
| `dashboard-ui/app/video-analysis/page.jsx`      | **Video Analysis Interface**     | Video upload, Temporal analysis, Event detection                 |
| `dashboard-ui/app/player-hub/page.jsx`          | **Player Management**            | Player profiles, Match history, Performance tracking             |
| `dashboard-ui/app/ai-coach-status/page.jsx`     | **System Status Dashboard**      | Health monitoring, Service status, Performance metrics           |

### **API Route Handlers**

| File                                    | Functionality               | Key Features                                             |
| --------------------------------------- | --------------------------- | -------------------------------------------------------- |
| `dashboard-ui/app/api/analyze/route.js` | **Frontend Analysis Proxy** | Request routing, Error handling, Response transformation |
| `dashboard-ui/app/api/health/route.js`  | **Health Check Proxy**      | Backend health monitoring, Service status                |
| `dashboard-ui/app/api/upload/route.js`  | **File Upload Handler**     | File validation, Local storage, Metadata management      |

### **UI Components**

| File                                     | Functionality                  | Key Features                                 |
| ---------------------------------------- | ------------------------------ | -------------------------------------------- |
| `dashboard-ui/components/ClientOnly.jsx` | **Hydration Safety Component** | SSR/CSR compatibility, Client-side rendering |
| `dashboard-ui/components/ui/*`           | **Reusable UI Components**     | Buttons, Cards, Forms, Modals, Charts        |

---

## 🧠 **CORE AI/ML ENGINE**

### **Primary Analysis Systems**

| File                                       | Functionality                      | Lines | Key Features                                                 |
| ------------------------------------------ | ---------------------------------- | ----- | ------------------------------------------------------------ |
| `core/ultimate_parsing_system.py`          | **Ultra-High Confidence Analysis** | 631   | 95-100% accuracy, Multi-stage processing, Quality validation |
| `core/enhanced_ultimate_parsing_system.py` | **Enhanced Ultimate System**       | 139   | Improved performance, Better error handling                  |
| `core/enhanced_data_collector.py`          | **Advanced Data Extraction**       | 2031  | OCR processing, Pattern recognition, Data validation         |
| `core/data_collector.py`                   | **Basic Data Collection**          | 596   | Screenshot analysis, Text extraction, Hero detection         |

### **Specialized Detectors**

| File                                 | Functionality                | Lines | Key Features                                               |
| ------------------------------------ | ---------------------------- | ----- | ---------------------------------------------------------- |
| `core/advanced_hero_detector.py`     | **Hero Recognition System**  | 289   | Portrait detection, Name matching, Confidence scoring      |
| `core/premium_hero_detector.py`      | **Premium Hero Detection**   | 910   | Advanced algorithms, Multi-method detection, High accuracy |
| `core/row_specific_hero_detector.py` | **Row-Based Hero Detection** | 327   | Position-aware detection, Team composition analysis        |
| `core/trophy_medal_detector_v2.py`   | **Award Detection System**   | 701   | MVP detection, Medal recognition, Achievement analysis     |

### **Performance & Quality Systems**

| File                                    | Functionality             | Lines | Key Features                                                    |
| --------------------------------------- | ------------------------- | ----- | --------------------------------------------------------------- |
| `core/advanced_performance_analyzer.py` | **Performance Analysis**  | 689   | KDA analysis, Gold efficiency, Impact scoring                   |
| `core/elite_confidence_scorer.py`       | **Confidence Assessment** | 603   | Reliability scoring, Quality metrics, Trust indicators          |
| `core/advanced_quality_validator.py`    | **Quality Validation**    | 405   | Data integrity, Accuracy validation, Error detection            |
| `core/intelligent_data_completer.py`    | **Smart Data Completion** | 1055  | Missing data inference, Pattern completion, Confidence boosting |

### **Behavioral & Temporal Analysis**

| File                          | Functionality                | Lines | Key Features                                                    |
| ----------------------------- | ---------------------------- | ----- | --------------------------------------------------------------- |
| `core/behavioral_modeling.py` | **Player Behavior Analysis** | 663   | Playstyle detection, Pattern recognition, Personality profiling |
| `core/temporal_pipeline.py`   | **Time-Series Analysis**     | 408   | Event sequencing, Timeline analysis, Temporal patterns          |
| `core/event_detector.py`      | **Game Event Recognition**   | 710   | Action detection, Event classification, Timing analysis         |
| `core/minimap_tracker.py`     | **Minimap Analysis**         | 606   | Position tracking, Movement patterns, Map awareness             |

### **Utility & Support Systems**

| File                        | Functionality            | Lines | Key Features                                       |
| --------------------------- | ------------------------ | ----- | -------------------------------------------------- |
| `core/hero_database.py`     | **Hero Data Management** | 452   | Hero metadata, Role mappings, Ability information  |
| `core/session_manager.py`   | **Session Management**   | 183   | User sessions, State management, Data persistence  |
| `core/error_handler.py`     | **Error Management**     | 535   | Exception handling, Error recovery, Logging        |
| `core/diagnostic_logger.py` | **System Diagnostics**   | 331   | Performance logging, Debug information, Monitoring |

---

## 🎮 **GAME RULES & LOGIC**

### **Hero-Specific Rules**

| File                               | Functionality                  | Key Features                                                        |
| ---------------------------------- | ------------------------------ | ------------------------------------------------------------------- |
| `rules/roles/tank/tigreal.py`      | **Tigreal Tank Analysis**      | Tank-specific metrics, Crowd control effectiveness, Team protection |
| `rules/roles/marksman/miya.py`     | **Miya Marksman Analysis**     | DPS optimization, Positioning analysis, Late-game scaling           |
| `rules/roles/assassin/lancelot.py` | **Lancelot Assassin Analysis** | Burst damage, Target selection, Mobility usage                      |
| `rules/roles/mage/kagura.py`       | **Kagura Mage Analysis**       | Skill combos, Magic damage, Zone control                            |
| `rules/roles/support/estes.py`     | **Estes Support Analysis**     | Healing efficiency, Team support, Vision control                    |

### **Core Rule System**

| File                      | Functionality                 | Key Features                                                 |
| ------------------------- | ----------------------------- | ------------------------------------------------------------ |
| `coach.py`                | **Main Coaching Engine**      | Rule discovery, Dynamic loading, Feedback generation         |
| `core/role_evaluators.py` | **Role-Based Evaluation**     | Role-specific analysis, Performance metrics, Recommendations |
| `core/base_evaluator.py`  | **Base Evaluation Framework** | Common evaluation logic, Metrics calculation, Scoring system |

---

## 📊 **DATA & CONFIGURATION**

### **Hero & Game Data**

| File                              | Functionality              | Size  | Key Features                                              |
| --------------------------------- | -------------------------- | ----- | --------------------------------------------------------- |
| `data/mlbb-heroes-corrected.json` | **Complete Hero Database** | 35KB  | Hero metadata, Abilities, Role mappings, Meta information |
| `data/heroes_cleaned.json`        | **Processed Hero Data**    | 105KB | Cleaned datasets, Standardized format, Analysis-ready     |
| `data/hero_role_mapping.json`     | **Role Classifications**   | 16KB  | Hero-to-role mappings, Team composition data              |

### **Configuration Files**

| File                             | Functionality           | Key Features                                           |
| -------------------------------- | ----------------------- | ------------------------------------------------------ |
| `config/enhanced_thresholds.yml` | **Advanced Thresholds** | Quality gates, Confidence levels, Performance criteria |
| `config/thresholds.yml`          | **Basic Thresholds**    | Standard configurations, Default values                |
| `requirements.txt`               | **Python Dependencies** | Package management, Version specifications             |

### **ML Models**

| File                       | Functionality            | Size                                             | Key Features                               |
| -------------------------- | ------------------------ | ------------------------------------------------ | ------------------------------------------ |
| `models/mlbb_yolo_best.pt` | **YOLO Detection Model** | Custom-trained, Object detection, Hero portraits |
| `yolov8n.pt`               | **Base YOLO Model**      | 6.2MB                                            | Foundation model, General object detection |

---

## 🔧 **CORE SERVICES**

### **Service Architecture**

| File                                           | Functionality                  | Lines | Key Features                                                       |
| ---------------------------------------------- | ------------------------------ | ----- | ------------------------------------------------------------------ |
| `core/services/analysis_service.py`            | **Main Analysis Orchestrator** | 340   | Service coordination, Request routing, Response aggregation        |
| `core/services/yolo_detection_service.py`      | **YOLO Integration Service**   | 454   | Object detection, Model inference, Result processing               |
| `core/services/tactical_coaching_service.py`   | **Tactical Analysis Service**  | 690   | Strategic insights, Coaching recommendations, Performance analysis |
| `core/services/behavioral_modeling_service.py` | **Behavior Analysis Service**  | 524   | Playstyle detection, Pattern analysis, Player profiling            |

### **Specialized Services**

| File                                            | Functionality                 | Lines | Key Features                                           |
| ----------------------------------------------- | ----------------------------- | ----- | ------------------------------------------------------ |
| `core/services/team_behavior_service.py`        | **Team Dynamics Service**     | 706   | Team coordination, Synergy analysis, Group performance |
| `core/services/hero_evaluation_orchestrator.py` | **Hero Evaluation Service**   | 618   | Hero performance, Role effectiveness, Meta analysis    |
| `core/services/ocr_service.py`                  | **OCR Processing Service**    | 214   | Text extraction, Character recognition, Data parsing   |
| `core/services/detection_service.py`            | **General Detection Service** | 195   | Pattern detection, Feature extraction, Classification  |

---

## 🧪 **TESTING & VALIDATION**

### **Test Categories**

| Type                  | Files                   | Purpose                                                   |
| --------------------- | ----------------------- | --------------------------------------------------------- |
| **Unit Tests**        | `test_*.py`             | Component testing, Function validation, Error handling    |
| **Integration Tests** | `test_*_integration.py` | System integration, API testing, End-to-end workflows     |
| **Performance Tests** | `test_performance_*.py` | Speed optimization, Memory usage, Scalability             |
| **Validation Tests**  | `test_*_validation.py`  | Accuracy testing, Quality assurance, Real-world scenarios |

### **Key Test Files**

| File                                 | Purpose                      | Key Features                                    |
| ------------------------------------ | ---------------------------- | ----------------------------------------------- |
| `test_ultimate_system.py`            | **Ultimate System Testing**  | High-confidence analysis, Quality validation    |
| `test_yolo_integration.py`           | **YOLO Integration Testing** | Object detection, Model performance             |
| `test_real_screenshot_validation.py` | **Real-World Testing**       | User screenshot validation, Accuracy assessment |
| `test_tactical_coaching_system.py`   | **Coaching System Testing**  | Strategic analysis, Recommendation quality      |

---

## 🚀 **DEPLOYMENT & UTILITIES**

### **Startup & Deployment**

| File                    | Functionality                | Key Features                                                     |
| ----------------------- | ---------------------------- | ---------------------------------------------------------------- |
| `start_all_services.sh` | **Complete Service Startup** | Backend + Frontend launch, Health monitoring, Process management |
| `deploy_yolo_model.py`  | **YOLO Model Deployment**    | Model optimization, Performance tuning, Production setup         |
| `diagnose_backend.sh`   | **Backend Diagnostics**      | Health checks, Performance analysis, Troubleshooting             |

### **Data Management**

| File                        | Functionality          | Key Features                                                         |
| --------------------------- | ---------------------- | -------------------------------------------------------------------- |
| `data/hero_data_cleaner.py` | **Data Preprocessing** | Data cleaning, Format standardization, Quality assurance             |
| `batch_yolo_inference.py`   | **Batch Processing**   | Large-scale analysis, Performance optimization, Resource management  |
| `train_yolo_detector.py`    | **Model Training**     | Custom model training, Dataset preparation, Performance optimization |

---

## 🎯 **ENTRY POINTS & MAIN FLOWS**

### **Primary Entry Points**

1. **`main.py`** → Command-line coaching analysis
2. **`web/app.py`** → FastAPI backend server
3. **`dashboard-ui/app/page.jsx`** → React frontend application
4. **`coach.py`** → Core coaching engine

### **Analysis Flow**

```
Upload → Frontend API → Backend Processing → AI Analysis → Results → Display
```

### **Key Data Flow**

```
Screenshot → OCR → Hero Detection → Performance Analysis → Coaching → Mental Boost
```

---

## 📈 **PERFORMANCE & OPTIMIZATION**

### **Speed Optimizations**

- **`/api/analyze-instant`** → 2-second response guarantee
- **`core/elite_confidence_scorer.py`** → Quality-based fast processing
- **`core/yolo_fallback.py`** → Intelligent fallback mechanisms
- **`core/cache/`** → Result caching and performance boosting

### **Quality Assurance**

- **`core/advanced_quality_validator.py`** → Multi-stage validation
- **`core/validation_manager.py`** → Quality control and user feedback
- **`core/edge_case_tester.py`** → Edge case handling and testing

---

## 🔧 **SYSTEM MONITORING**

### **Health & Diagnostics**

- **`/api/health-isolated`** → Never-blocking health check
- **`core/diagnostic_logger.py`** → Comprehensive system logging
- **`core/performance_monitor.py`** → Real-time performance tracking
- **`web/debug_panel.py`** → Live system diagnostics

### **Error Handling**

- **`core/error_handler.py`** → Centralized error management
- **`core/realtime_confidence_adjuster.py`** → Dynamic quality adjustment
- **Timeout mechanisms** → Prevent system hanging

---

## 🎮 **GAME-SPECIFIC FEATURES**

### **MLBB-Specific Components**

- **MVP/Medal Detection** → Achievement recognition
- **Hero Database** → Complete hero metadata
- **Role-Based Analysis** → Position-specific evaluation
- **Team Composition** → Synergy and counter-pick analysis
- **Meta Analysis** → Current game meta integration

### **Advanced Features**

- **Behavioral Modeling** → Playstyle detection and analysis
- **Temporal Analysis** → Time-based performance tracking
- **Video Analysis** → Frame-by-frame gameplay analysis
- **Tactical Coaching** → Strategic improvement recommendations

---

## 📋 **DOCUMENTATION FILES**

| File                                 | Purpose                           |
| ------------------------------------ | --------------------------------- |
| `FRONTEND_FIXES_SUMMARY.md`          | Frontend issue resolution         |
| `INTEGRATION_SUCCESS_REPORT.md`      | System integration status         |
| `BEHAVIORAL_MODELING_SYSTEM.md`      | Behavioral analysis documentation |
| `TACTICAL_COACHING_SYSTEM_README.md` | Tactical coaching guide           |
| `YOLO_INTEGRATION_GUIDE.md`          | YOLO implementation guide         |
| `REAL_USER_VALIDATION_GUIDE.md`      | User testing procedures           |

---

## 🎯 **SUMMARY**

The MLBB Coach AI project is a comprehensive, full-stack application featuring:

- **57KB+ FastAPI backend** with multiple specialized endpoints
- **Complete Next.js frontend** with modern React components
- **Advanced AI/ML pipeline** with 95-100% confidence systems
- **Extensive hero database** with 129+ heroes and metadata
- **Real-time analysis** with sub-5-second response times
- **Comprehensive testing** with validation and performance monitoring
- **Production-ready deployment** with health monitoring and error handling

**Total codebase: 100+ vital files, 50,000+ lines of code, full-stack AI coaching platform**

---

_Generated: 2025-01-15 | Version: 2.0.0 | Status: Production Ready_

## Git-Style Patches

### 1. Frontend Fix (`dashboard-ui/app/api/analyze/route.js`)

```diff
--- a/skillshift-ai/dashboard-ui/app/api/analyze/route.js
+++ b/skillshift-ai/dashboard-ui/app/api/analyze/route.js
@@ -6,6 +6,7 @@ import fetch from "node-fetch";
 const SKILLSHIFT_AI_BASE_URL =
   process.env.SKILLSHIFT_AI_URL || "http://localhost:8000";
 const DEFAULT_IGN = process.env.DEFAULT_IGN || "Lesz XVII";
+const ANALYZE_ENDPOINT = process.env.NEXT_PUBLIC_ANALYZE_ENDPOINT || "/api/analyze";
 const USE_FAST_ANALYSIS = process.env.USE_FAST_ANALYSIS === "true";

 // Experience stage configurations for enhanced feedback
@@ -103,10 +104,9 @@ export async function POST(request) {
       contentType: file.type,
     });

-    // Choose endpoint based on configuration - prioritize instant analysis
-    const endpoint = USE_FAST_ANALYSIS
-      ? "/api/analyze-instant"
-      : "/api/analyze-instant";
-    const timeout = 5000; // 5 second timeout for instant analysis
+    // Choose endpoint based on configuration - use env var or default to heavy analysis
+    const endpoint = USE_FAST_ANALYSIS ? "/api/analyze-instant" : ANALYZE_ENDPOINT;
+    const timeout = USE_FAST_ANALYSIS ? 5000 : 30000; // 5s for instant, 30s for heavy analysis

     // Call the skillshift-ai analysis endpoint with timeout
     let analysisResponse;
```

### 2. Backend Fix (`web/app.py`)

```diff
<code_block_to_apply_changes_from>
```

## Verification Steps

• **Test default routing**: `curl -X POST http://localhost:3000/api/analyze -F "file=@test.png" | jq '.analysis.performance.hero'` → Should return actual hero name, not "Unknown"

• **Test fast mode**: `USE_FAST_ANALYSIS=true curl -X POST http://localhost:3000/api/analyze -F "file=@test.png" | jq '.job_enqueued'` → Should return `true` or `false` depending on RQ availability

• **Check backend direct**: `curl -X POST http://localhost:8000/api/analyze -F "file=@test.png" | jq '.overall_confidence'` → Should return >0 confidence score

• **Monitor logs**: `tail -f skillshift-ai/logs/app.log` → Should show "Ultimate Parsing System" processing, not stub responses

• **Verify env override**: `NEXT_PUBLIC_ANALYZE_ENDPOINT=/api/analyze-fast curl -X POST http://localhost:3000/api/analyze -F "file=@test.png"` → Should route to custom endpoint
