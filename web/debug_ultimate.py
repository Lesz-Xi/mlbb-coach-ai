"""
Debug Ultimate Parsing System Endpoint

This module provides a debug endpoint specifically for the Ultimate Parsing System
to showcase the 95-100% confidence capabilities.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from tempfile import NamedTemporaryFile
import shutil
import os
import logging

from core.ultimate_parsing_system import ultimate_parsing_system

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/debug/ultimate-analysis/")
async def debug_ultimate_analysis(
    file: UploadFile = File(...),
    ign: str = "Lesz XVII",
    quality_threshold: float = 85.0
):
    """
    Debug endpoint for Ultimate Parsing System with comprehensive analysis.
    
    Args:
        file: Screenshot file
        ign: Player's IGN
        quality_threshold: Minimum quality threshold (0-100)
        
    Returns:
        Complete ultimate analysis result
    """
    # Save uploaded file
    try:
        with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
    finally:
        file.file.close()

    try:
        # Run Ultimate Parsing System
        ultimate_result = ultimate_parsing_system.analyze_screenshot_ultimate(
            image_path=temp_file_path,
            ign=ign,
            hero_override=None,
            context="scoreboard",
            quality_threshold=quality_threshold
        )
        
        # Convert to JSON-serializable format
        result = {
            "status": "success",
            "confidence": {
                "overall": ultimate_result.overall_confidence,
                "category": ultimate_result.confidence_breakdown.category.value,
                "component_scores": ultimate_result.confidence_breakdown.component_scores,
                "quality_factors": ultimate_result.confidence_breakdown.quality_factors
            },
            "data": ultimate_result.parsed_data,
            "quality_assessment": {
                "overall_score": ultimate_result.quality_assessment.overall_score,
                "is_acceptable": ultimate_result.quality_assessment.is_acceptable,
                "issues": [issue.value for issue in ultimate_result.quality_assessment.issues],
                "recommendations": ultimate_result.quality_assessment.recommendations
            },
            "hero_detection": {
                "hero_name": ultimate_result.hero_detection.hero_name,
                "confidence": ultimate_result.hero_detection.confidence,
                "detection_method": ultimate_result.hero_detection.detection_method,
                "portrait_confidence": ultimate_result.hero_detection.portrait_confidence,
                "text_confidence": ultimate_result.hero_detection.text_confidence
            },
            "data_completion": {
                "completeness_score": ultimate_result.data_completion.completeness_score,
                "confidence_score": ultimate_result.data_completion.confidence_score,
                "methods_used": ultimate_result.data_completion.completion_methods
            },
            "performance": {
                "processing_time": ultimate_result.processing_time,
                "analysis_stage": ultimate_result.analysis_stage,
                "success_factors": ultimate_result.success_factors,
                "improvement_roadmap": ultimate_result.improvement_roadmap
            },
            "diagnostics": ultimate_result.diagnostic_info,
            "warnings": ultimate_result.warnings
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Ultimate analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Ultimate analysis failed: {str(e)}"
        )
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@router.get("/debug/ultimate-dashboard/")
async def debug_ultimate_dashboard():
    """
    Render the Ultimate Parsing System debug dashboard.
    """
    # For now, return a simple HTML response
    # In a real implementation, this would render the debug_dashboard.html template
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ultimate Parsing System - Debug Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { text-align: center; margin-bottom: 30px; }
            .upload-form { max-width: 500px; margin: 0 auto; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; }
            input[type="file"], input[type="text"], input[type="number"] { 
                width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; 
            }
            .submit-btn { 
                background: #007bff; color: white; padding: 12px 30px; 
                border: none; border-radius: 5px; cursor: pointer; 
            }
            .submit-btn:hover { background: #0056b3; }
            .result { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 Ultimate Parsing System</h1>
            <p>95-100% Confidence AI Coaching Debug Dashboard</p>
        </div>
        
        <div class="upload-form">
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="screenshot">Screenshot File:</label>
                    <input type="file" id="screenshot" name="file" accept="image/*" required>
                </div>
                
                <div class="form-group">
                    <label for="ign">Player IGN:</label>
                    <input type="text" id="ign" name="ign" value="Lesz XVII" required>
                </div>
                
                <div class="form-group">
                    <label for="quality_threshold">Quality Threshold (%):</label>
                    <input type="number" id="quality_threshold" name="quality_threshold" 
                           value="85" min="0" max="100" step="1">
                </div>
                
                <button type="submit" class="submit-btn">🔍 Analyze with Ultimate System</button>
            </form>
        </div>
        
        <div id="result" class="result" style="display: none;"></div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const resultDiv = document.getElementById('result');
                
                resultDiv.innerHTML = '<p>🔄 Analyzing with Ultimate Parsing System...</p>';
                resultDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/debug/ultimate-analysis/', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.status === 'success') {
                        resultDiv.innerHTML = `
                            <h3>✅ Ultimate Analysis Complete</h3>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                                <div>
                                    <h4>🎯 Confidence Results</h4>
                                    <p><strong>Overall Confidence:</strong> ${result.confidence.overall.toFixed(1)}%</p>
                                    <p><strong>Category:</strong> ${result.confidence.category.toUpperCase()}</p>
                                    <p><strong>Processing Time:</strong> ${result.performance.processing_time.toFixed(2)}s</p>
                                </div>
                                <div>
                                    <h4>📊 Component Scores</h4>
                                    ${Object.entries(result.confidence.component_scores).map(([key, value]) => 
                                        `<p><strong>${key.replace('_', ' ').toUpperCase()}:</strong> ${value.toFixed(1)}%</p>`
                                    ).join('')}
                                </div>
                            </div>
                            
                            <div style="margin-top: 20px;">
                                <h4>🎮 Extracted Data</h4>
                                <pre style="background: #f0f0f0; padding: 10px; border-radius: 5px; overflow-x: auto;">
${JSON.stringify(result.data, null, 2)}
                                </pre>
                            </div>
                            
                            ${result.performance.success_factors.length > 0 ? `
                                <div style="margin-top: 20px;">
                                    <h4>🏆 Success Factors</h4>
                                    <ul>
                                        ${result.performance.success_factors.map(factor => `<li>${factor}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                            
                            ${result.performance.improvement_roadmap.length > 0 ? `
                                <div style="margin-top: 20px;">
                                    <h4>🛠️ Improvement Roadmap</h4>
                                    <ul>
                                        ${result.performance.improvement_roadmap.map(item => `<li>${item}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                        `;
                    } else {
                        resultDiv.innerHTML = `<p style="color: red;">❌ Analysis failed: ${result.detail}</p>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<p style="color: red;">❌ Error: ${error.message}</p>`;
                }
            });
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)


@router.get("/debug/performance-summary/")
async def debug_performance_summary():
    """
    Get performance summary of the Ultimate Parsing System.
    """
    try:
        summary = ultimate_parsing_system.get_performance_summary()
        return {
            "status": "success",
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Performance summary failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Performance summary failed: {str(e)}"
        )