from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
import json
import base64
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

debug_router = APIRouter(prefix="/debug", tags=["debug"])

@debug_router.get("/panel", response_class=HTMLResponse)
async def debug_panel():
    """Serve the debug panel HTML interface."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SkillShift AI - Developer Debug Panel</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', system-ui, sans-serif; 
                background: #0a0e1a; 
                color: #e2e8f0; 
                line-height: 1.6;
            }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            
            .header {
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 30px;
                border: 1px solid #475569;
            }
            .header h1 { color: #fbbf24; font-size: 1.8rem; margin-bottom: 10px; }
            .header p { color: #94a3b8; }
            
            .controls {
                display: flex; 
                gap: 15px; 
                margin-bottom: 30px; 
                flex-wrap: wrap;
                align-items: center;
            }
            .control-group { display: flex; align-items: center; gap: 10px; }
            .control-group label { color: #cbd5e1; font-weight: 500; }
            
            .toggle-switch {
                position: relative;
                width: 60px;
                height: 30px;
                background: #475569;
                border-radius: 15px;
                cursor: pointer;
                transition: background 0.3s;
            }
            .toggle-switch.active { background: #10b981; }
            .toggle-switch::before {
                content: '';
                position: absolute;
                width: 26px;
                height: 26px;
                border-radius: 50%;
                background: white;
                top: 2px;
                left: 2px;
                transition: transform 0.3s;
            }
            .toggle-switch.active::before { transform: translateX(30px); }
            
            .session-selector select {
                background: #374151;
                color: #e2e8f0;
                border: 1px solid #6b7280;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
            }
            
            .main-content {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 30px;
            }
            
            .panel {
                background: #1e293b;
                border-radius: 12px;
                padding: 20px;
                border: 1px solid #334155;
            }
            .panel h2 {
                color: #fbbf24;
                margin-bottom: 15px;
                font-size: 1.2rem;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .image-container {
                position: relative;
                margin-bottom: 20px;
                border-radius: 8px;
                overflow: hidden;
                background: #0f172a;
            }
            .image-container img {
                width: 100%;
                height: auto;
                display: block;
            }
            .overlay-toggle {
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(0,0,0,0.7);
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 12px;
            }
            
            .data-section {
                margin-bottom: 20px;
            }
            .data-section h3 {
                color: #38bdf8;
                margin-bottom: 10px;
                font-size: 1rem;
            }
            
            .ocr-results {
                max-height: 300px;
                overflow-y: auto;
                background: #0f172a;
                border-radius: 6px;
                padding: 15px;
                border: 1px solid #374151;
            }
            .ocr-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 0;
                border-bottom: 1px solid #374151;
            }
            .ocr-item:last-child { border-bottom: none; }
            .ocr-text { 
                font-family: 'Courier New', monospace; 
                color: #e2e8f0;
                flex: 1;
            }
            .ocr-confidence {
                background: #374151;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                margin-left: 10px;
            }
            .confidence-high { background: #10b981; }
            .confidence-medium { background: #f59e0b; }
            .confidence-low { background: #ef4444; }
            .ocr-category {
                background: #6366f1;
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 10px;
                margin-left: 5px;
            }
            
            .analysis-results {
                background: #0f172a;
                border-radius: 6px;
                padding: 15px;
                border: 1px solid #374151;
            }
            .result-item {
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
                padding: 5px 0;
            }
            .result-key { color: #94a3b8; }
            .result-value { 
                color: #e2e8f0; 
                font-weight: 500;
                font-family: 'Courier New', monospace;
            }
            
            .diagnostics-panel {
                grid-column: 1 / -1;
                background: #1e293b;
                border-radius: 12px;
                padding: 20px;
                border: 1px solid #334155;
            }
            
            .diagnostics-steps {
                display: grid;
                gap: 15px;
            }
            .step {
                background: #0f172a;
                border-radius: 8px;
                padding: 15px;
                border-left: 4px solid #6366f1;
            }
            .step.warning { border-left-color: #f59e0b; }
            .step.error { border-left-color: #ef4444; }
            .step.success { border-left-color: #10b981; }
            
            .step-header {
                display: flex;
                justify-content: between;
                align-items: center;
                margin-bottom: 10px;
            }
            .step-name { 
                color: #fbbf24; 
                font-weight: 600;
                flex: 1;
            }
            .step-confidence {
                background: #374151;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
            }
            .step-time {
                color: #94a3b8;
                font-size: 12px;
                margin-left: 10px;
            }
            
            .step-details {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-top: 10px;
            }
            .detail-section h4 {
                color: #38bdf8;
                margin-bottom: 5px;
                font-size: 0.9rem;
            }
            .detail-content {
                background: #1e293b;
                padding: 10px;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                max-height: 150px;
                overflow-y: auto;
            }
            
            .warnings-errors {
                margin-top: 10px;
            }
            .warning-item, .error-item {
                padding: 5px 10px;
                border-radius: 4px;
                margin-bottom: 5px;
                font-size: 13px;
            }
            .warning-item { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
            .error-item { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }
            
            .failure-summary {
                background: #7c2d12;
                border: 1px solid #dc2626;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
            }
            .failure-summary h3 { color: #fca5a5; margin-bottom: 10px; }
            .recommendation {
                background: rgba(34, 197, 94, 0.1);
                border: 1px solid #22c55e;
                padding: 8px 12px;
                border-radius: 6px;
                margin: 5px 0;
                color: #86efac;
                font-size: 13px;
            }
            
            .simulation-controls {
                background: #7c2d12;
                border: 1px solid #dc2626;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
            }
            .simulation-controls h3 { color: #fca5a5; margin-bottom: 10px; }
            .sim-button {
                background: #dc2626;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                margin: 5px;
                font-size: 12px;
            }
            .sim-button:hover { background: #b91c1c; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔧 SkillShift AI - Developer Debug Panel</h1>
                <p>Raw OCR Analysis vs Final System Output • Real-time Diagnostic Visualization</p>
            </div>
            
            <div class="controls">
                <div class="control-group">
                    <label>Show OCR Overlays</label>
                    <div class="toggle-switch active" id="overlayToggle"></div>
                </div>
                <div class="control-group">
                    <label>Live Diagnostics</label>
                    <div class="toggle-switch active" id="liveToggle"></div>
                </div>
                <div class="control-group">
                    <label>Session:</label>
                    <div class="session-selector">
                        <select id="sessionSelect">
                            <option value="latest">Latest Analysis</option>
                        </select>
                    </div>
                </div>
                <button class="sim-button" onclick="loadLatestSession()">🔄 Refresh</button>
            </div>
            
            <div id="failureSummary" class="failure-summary" style="display: none;">
                <h3>⚠️ Analysis Failed - Debugging Information</h3>
                <div id="failureDetails"></div>
                <div id="recommendations"></div>
            </div>
            
            <div class="simulation-controls">
                <h3>🧪 Simulation Controls (for testing error handling)</h3>
                <button class="sim-button" onclick="simulateFailure('hero_detection')">Simulate Hero Detection Failure</button>
                <button class="sim-button" onclick="simulateFailure('ocr_low_confidence')">Simulate Low OCR Confidence</button>
                <button class="sim-button" onclick="simulateFailure('gold_parsing')">Simulate Gold Parsing Failure</button>
                <button class="sim-button" onclick="simulateFailure('complete_failure')">Simulate Complete Analysis Failure</button>
            </div>
            
            <div class="main-content">
                <div class="panel">
                    <h2>📷 Raw Screenshot + OCR Overlays</h2>
                    <div class="image-container">
                        <img id="originalImage" src="/api/placeholder-image" alt="Screenshot">
                        <img id="overlayImage" src="/api/placeholder-overlay" alt="OCR Overlay" style="position: absolute; top: 0; left: 0; opacity: 0.8; display: none;">
                        <button class="overlay-toggle" onclick="toggleOverlay()">Toggle Overlay</button>
                    </div>
                    
                    <div class="data-section">
                        <h3>🔍 Raw OCR Detections</h3>
                        <div class="ocr-results" id="ocrResults">
                            <div class="ocr-item">
                                <span class="ocr-text">Loading OCR data...</span>
                                <span class="ocr-confidence">--</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="panel">
                    <h2>🎯 Final Analysis Output</h2>
                    <div class="data-section">
                        <h3>📊 Parsed Game Data</h3>
                        <div class="analysis-results" id="analysisResults">
                            <div class="result-item">
                                <span class="result-key">Loading...</span>
                                <span class="result-value">--</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="data-section">
                        <h3>🎮 Hero Detection Results</h3>
                        <div class="analysis-results" id="heroResults">
                            <div class="result-item">
                                <span class="result-key">Hero:</span>
                                <span class="result-value">Loading...</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="data-section">
                        <h3>📈 Confidence Metrics</h3>
                        <div class="analysis-results" id="confidenceResults">
                            <div class="result-item">
                                <span class="result-key">Overall:</span>
                                <span class="result-value">Loading...</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="diagnostics-panel">
                <h2>🔬 Detailed Processing Diagnostics</h2>
                <div class="diagnostics-steps" id="diagnosticsSteps">
                    <div class="step">
                        <div class="step-header">
                            <span class="step-name">Loading diagnostic data...</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let currentSession = null;
            let showOverlay = true;
            let liveDiagnostics = true;
            
            function toggleOverlay() {
                const overlay = document.getElementById('overlayImage');
                showOverlay = !showOverlay;
                overlay.style.display = showOverlay ? 'block' : 'none';
            }
            
            function initializeToggles() {
                document.getElementById('overlayToggle').addEventListener('click', function() {
                    this.classList.toggle('active');
                    toggleOverlay();
                });
                
                document.getElementById('liveToggle').addEventListener('click', function() {
                    this.classList.toggle('active');
                    liveDiagnostics = this.classList.contains('active');
                    if (liveDiagnostics) {
                        startLiveUpdates();
                    } else {
                        stopLiveUpdates();
                    }
                });
            }
            
            async function loadLatestSession() {
                try {
                    const response = await fetch('/debug/latest-session');
                    const data = await response.json();
                    
                    if (data.error) {
                        displayError(data.error);
                        return;
                    }
                    
                    currentSession = data;
                    updateInterface(data);
                    
                } catch (error) {
                    console.error('Failed to load session:', error);
                    displayError('Failed to load latest session');
                }
            }
            
            function updateInterface(sessionData) {
                // Update images
                if (sessionData.image_path) {
                    document.getElementById('originalImage').src = `/debug/image/${sessionData.session_id}`;
                }
                if (sessionData.overlay_image_base64) {
                    document.getElementById('overlayImage').src = `data:image/png;base64,${sessionData.overlay_image_base64}`;
                }
                
                // Update OCR results
                updateOCRResults(sessionData.steps || []);
                
                // Update analysis results
                updateAnalysisResults(sessionData.final_data || {});
                
                // Update hero detection
                updateHeroResults(sessionData.hero_info || {});
                
                // Update confidence metrics
                updateConfidenceResults(sessionData);
                
                // Update diagnostics
                updateDiagnostics(sessionData.steps || []);
                
                // Show failure summary if needed
                updateFailureSummary(sessionData);
            }
            
            function updateOCRResults(steps) {
                const container = document.getElementById('ocrResults');
                container.innerHTML = '';
                
                steps.forEach(step => {
                    step.ocr_detections?.forEach(detection => {
                        const item = document.createElement('div');
                        item.className = 'ocr-item';
                        
                        const confidenceClass = detection.confidence > 0.8 ? 'confidence-high' : 
                                              detection.confidence > 0.5 ? 'confidence-medium' : 'confidence-low';
                        
                        item.innerHTML = `
                            <span class="ocr-text">"${detection.text}"</span>
                            <span class="ocr-category">${detection.category}</span>
                            <span class="ocr-confidence ${confidenceClass}">${(detection.confidence * 100).toFixed(0)}%</span>
                        `;
                        container.appendChild(item);
                    });
                });
            }
            
            function updateAnalysisResults(data) {
                const container = document.getElementById('analysisResults');
                container.innerHTML = '';
                
                const importantFields = ['kills', 'deaths', 'assists', 'gold', 'hero_damage', 'match_result'];
                importantFields.forEach(field => {
                    if (data[field] !== undefined) {
                        const item = document.createElement('div');
                        item.className = 'result-item';
                        item.innerHTML = `
                            <span class="result-key">${field}:</span>
                            <span class="result-value">${data[field]}</span>
                        `;
                        container.appendChild(item);
                    }
                });
            }
            
            function updateHeroResults(heroInfo) {
                const container = document.getElementById('heroResults');
                container.innerHTML = `
                    <div class="result-item">
                        <span class="result-key">Detected Hero:</span>
                        <span class="result-value">${heroInfo.name || 'Unknown'}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-key">Confidence:</span>
                        <span class="result-value">${((heroInfo.confidence || 0) * 100).toFixed(1)}%</span>
                    </div>
                    <div class="result-item">
                        <span class="result-key">Strategies Tried:</span>
                        <span class="result-value">${(heroInfo.strategies_tried || []).join(', ')}</span>
                    </div>
                `;
            }
            
            function updateConfidenceResults(sessionData) {
                const container = document.getElementById('confidenceResults');
                container.innerHTML = `
                    <div class="result-item">
                        <span class="result-key">Overall:</span>
                        <span class="result-value">${((sessionData.final_confidence || 0) * 100).toFixed(1)}%</span>
                    </div>
                    <div class="result-item">
                        <span class="result-key">Data Completeness:</span>
                        <span class="result-value">${((sessionData.completeness_score || 0) * 100).toFixed(1)}%</span>
                    </div>
                    <div class="result-item">
                        <span class="result-key">Analysis Mode:</span>
                        <span class="result-value">${sessionData.analysis_mode || 'Unknown'}</span>
                    </div>
                `;
            }
            
            function updateDiagnostics(steps) {
                const container = document.getElementById('diagnosticsSteps');
                container.innerHTML = '';
                
                steps.forEach(step => {
                    const stepDiv = document.createElement('div');
                    const stepClass = step.errors?.length > 0 ? 'error' : 
                                    step.warnings?.length > 0 ? 'warning' : 'success';
                    stepDiv.className = `step ${stepClass}`;
                    
                    const confidenceClass = step.confidence_score > 0.7 ? 'confidence-high' : 
                                          step.confidence_score > 0.4 ? 'confidence-medium' : 'confidence-low';
                    
                    stepDiv.innerHTML = `
                        <div class="step-header">
                            <span class="step-name">${step.step_name}</span>
                            <span class="step-confidence ${confidenceClass}">${(step.confidence_score * 100).toFixed(0)}%</span>
                            <span class="step-time">${step.processing_time_ms.toFixed(0)}ms</span>
                        </div>
                        <div class="step-details">
                            <div class="detail-section">
                                <h4>Input Data</h4>
                                <div class="detail-content">${JSON.stringify(step.input_data, null, 2)}</div>
                            </div>
                            <div class="detail-section">
                                <h4>Output Data</h4>
                                <div class="detail-content">${JSON.stringify(step.output_data, null, 2)}</div>
                            </div>
                        </div>
                        <div class="warnings-errors">
                            ${step.warnings?.map(w => `<div class="warning-item">⚠️ ${w}</div>`).join('') || ''}
                            ${step.errors?.map(e => `<div class="error-item">❌ ${e}</div>`).join('') || ''}
                        </div>
                    `;
                    container.appendChild(stepDiv);
                });
            }
            
            function updateFailureSummary(sessionData) {
                const summary = sessionData.failure_summary;
                const container = document.getElementById('failureSummary');
                
                if (!summary || sessionData.final_confidence > 0.7) {
                    container.style.display = 'none';
                    return;
                }
                
                container.style.display = 'block';
                
                const details = document.getElementById('failureDetails');
                const recommendations = document.getElementById('recommendations');
                
                let detailsHTML = '';
                if (summary.hero_detection_failed) detailsHTML += '❌ Hero detection failed<br>';
                if (summary.gold_parsing_failed) detailsHTML += '❌ Gold/economy parsing failed<br>';
                if (summary.low_ocr_confidence) detailsHTML += '❌ Low OCR confidence<br>';
                if (summary.insufficient_data) detailsHTML += '❌ Insufficient data extracted<br>';
                
                details.innerHTML = detailsHTML;
                
                const recsHTML = summary.recommendations?.map(rec => 
                    `<div class="recommendation">💡 ${rec}</div>`
                ).join('') || '';
                recommendations.innerHTML = recsHTML;
            }
            
            function simulateFailure(type) {
                // This would call an API endpoint to simulate different failure scenarios
                console.log(`Simulating ${type} failure`);
                alert(`Simulating ${type} - this would trigger test scenarios for debugging error handling`);
            }
            
            function displayError(message) {
                console.error(message);
                // Display error in UI
            }
            
            let liveUpdateInterval;
            
            function startLiveUpdates() {
                if (liveUpdateInterval) clearInterval(liveUpdateInterval);
                liveUpdateInterval = setInterval(loadLatestSession, 2000); // Update every 2 seconds
            }
            
            function stopLiveUpdates() {
                if (liveUpdateInterval) {
                    clearInterval(liveUpdateInterval);
                    liveUpdateInterval = null;
                }
            }
            
            // Initialize the interface
            document.addEventListener('DOMContentLoaded', function() {
                initializeToggles();
                loadLatestSession();
                startLiveUpdates();
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@debug_router.get("/latest-session")
async def get_latest_session():
    """Get the latest diagnostic session data."""
    try:
        # Get the most recent diagnostic report
        debug_dir = Path("temp/diagnostics")
        if not debug_dir.exists():
            return {"error": "No diagnostic sessions found"}
        
        report_files = list(debug_dir.glob("*_report.json"))
        if not report_files:
            return {"error": "No diagnostic reports found"}
        
        # Get the most recent report
        latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_report, 'r') as f:
            diagnostics = json.load(f)
        
        # Process the data for frontend consumption
        return format_diagnostics_for_frontend(diagnostics)
        
    except Exception as e:
        logger.error(f"Failed to get latest session: {e}")
        return {"error": str(e)}

@debug_router.get("/image/{session_id}")
async def get_session_image(session_id: str):
    """Get the original image for a session."""
    try:
        debug_dir = Path("temp/diagnostics")
        overlay_file = debug_dir / f"{session_id}_overlay.png"
        
        if overlay_file.exists():
            with open(overlay_file, 'rb') as f:
                image_data = f.read()
            
            from fastapi.responses import Response
            return Response(content=image_data, media_type="image/png")
        else:
            # Return placeholder image
            return {"error": "Image not found"}
            
    except Exception as e:
        logger.error(f"Failed to get session image: {e}")
        return {"error": str(e)}

def format_diagnostics_for_frontend(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """Format diagnostic data for frontend consumption."""
    try:
        # Extract key information
        steps = diagnostics.get("steps", [])
        
        # Find hero detection step
        hero_info = {}
        for step in steps:
            if step.get("step_name") == "Hero_Detection":
                hero_info = {
                    "name": step.get("output_data", {}).get("hero", "Unknown"),
                    "confidence": step.get("confidence_score", 0),
                    "strategies_tried": step.get("output_data", {}).get("strategies_tried", [])
                }
                break
        
        # Extract final parsed data
        final_data = {}
        for step in steps:
            if step.get("step_name") == "Data_Parsing":
                final_data = step.get("output_data", {})
                break
        
        # Generate failure summary
        failure_summary = generate_failure_summary(diagnostics)
        
        return {
            "session_id": diagnostics.get("session_id"),
            "image_path": diagnostics.get("image_path"),
            "analysis_mode": diagnostics.get("analysis_mode"),
            "timestamp": diagnostics.get("timestamp"),
            "steps": steps,
            "final_confidence": diagnostics.get("final_confidence", 0),
            "final_warnings": diagnostics.get("final_warnings", []),
            "final_errors": diagnostics.get("final_errors", []),
            "overlay_image_base64": diagnostics.get("overlay_image_base64"),
            "hero_info": hero_info,
            "final_data": final_data,
            "completeness_score": get_completeness_score(steps),
            "failure_summary": failure_summary
        }
        
    except Exception as e:
        logger.error(f"Failed to format diagnostics: {e}")
        return {"error": "Failed to format diagnostic data"}

def generate_failure_summary(diagnostics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate a failure summary for debugging."""
    final_confidence = diagnostics.get("final_confidence", 0)
    
    if final_confidence > 0.7:
        return None  # No failure
    
    summary = {
        "hero_detection_failed": False,
        "gold_parsing_failed": False,
        "low_ocr_confidence": False,
        "insufficient_data": False,
        "recommendations": []
    }
    
    steps = diagnostics.get("steps", [])
    
    for step in steps:
        step_name = step.get("step_name", "")
        confidence = step.get("confidence_score", 0)
        errors = step.get("errors", [])
        
        if "Hero" in step_name and confidence < 0.7:
            summary["hero_detection_failed"] = True
            summary["recommendations"].append("Try a clearer screenshot with visible hero portraits")
        
        if "Data_Parsing" in step_name and any("gold" in err.lower() for err in errors):
            summary["gold_parsing_failed"] = True
            summary["recommendations"].append("Ensure gold values are clearly visible")
        
        if "OCR" in step_name and confidence < 0.8:
            summary["low_ocr_confidence"] = True
            summary["recommendations"].append("Upload higher resolution screenshots")
    
    if final_confidence < 0.3:
        summary["insufficient_data"] = True
        summary["recommendations"].append("Try uploading both scoreboard and stats screenshots")
    
    return summary

def get_completeness_score(steps: List[Dict[str, Any]]) -> float:
    """Extract completeness score from steps."""
    for step in steps:
        if step.get("step_name") == "Data_Parsing":
            return step.get("output_data", {}).get("completeness", 0)
    return 0 