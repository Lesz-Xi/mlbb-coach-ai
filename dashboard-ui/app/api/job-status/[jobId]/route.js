import { NextResponse } from "next/server";
import fetch from "node-fetch";

// Configuration for the skillshift-ai backend
const SKILLSHIFT_AI_BASE_URL =
  process.env.SKILLSHIFT_AI_URL || "http://localhost:8000";

// Experience stage configurations for enhanced feedback
const EXPERIENCE_STAGES = {
  first_upload: {
    anticipationMessage: "🧠 Decoding your first gameplay DNA...",
    revelationPrefix: "🎯 Your gameplay soul revealed:",
  },
  mvp_revealed: {
    anticipationMessage: "🎮 Analyzing your tactical instincts...",
    revelationPrefix: "⚡ Your decision patterns unlocked:",
  },
  deeper_insights: {
    anticipationMessage: "📈 Mapping your growth trajectory...",
    revelationPrefix: "🚀 Your mastery evolution:",
  },
  mastery_journey: {
    anticipationMessage: "🏆 Revealing your competitive edge...",
    revelationPrefix: "⭐ Elite insights unlocked:",
  },
};

export async function GET(request, { params }) {
  try {
    const jobId = params.jobId;

    if (!jobId) {
      return NextResponse.json(
        { success: false, error: "Job ID is required" },
        { status: 400 }
      );
    }

    // Get job status from backend
    const statusResponse = await fetch(
      `${SKILLSHIFT_AI_BASE_URL}/api/jobs/${jobId}`,
      {
        method: "GET",
        signal: AbortSignal.timeout(5000), // 5 second timeout
      }
    );

    if (!statusResponse.ok) {
      if (statusResponse.status === 404) {
        return NextResponse.json(
          { success: false, error: "Job not found" },
          { status: 404 }
        );
      }

      const errorText = await statusResponse.text();
      return NextResponse.json(
        {
          success: false,
          error: "Failed to get job status",
          details: errorText,
        },
        { status: statusResponse.status }
      );
    }

    const statusResult = await statusResponse.json();

    // If job is still running, return status info
    if (statusResult.state !== "finished") {
      return NextResponse.json({
        success: true,
        job_id: jobId,
        state: statusResult.state,
        message: statusResult.message || `Job is ${statusResult.state}`,
        is_complete: false,
        progress: getProgressMessage(statusResult.state),
      });
    }

    // Job is finished - transform results for dashboard
    if (statusResult.state === "failed") {
      return NextResponse.json({
        success: false,
        job_id: jobId,
        state: "failed",
        error: statusResult.error || "Analysis failed",
        is_complete: true,
      });
    }

    // Job completed successfully - transform the result
    const analysisResult = statusResult.result;

    if (!analysisResult || !analysisResult.success) {
      return NextResponse.json({
        success: false,
        job_id: jobId,
        state: "failed",
        error: analysisResult?.error || "Analysis produced no results",
        is_complete: true,
      });
    }

    // Transform to match dashboard expectations
    const dashboardResponse = {
      success: true,
      job_id: jobId,
      state: "finished",
      is_complete: true,
      message: "Analysis completed successfully",
      analysis: {
        // Core analysis results
        statistical_feedback: analysisResult.feedback || [],
        mental_feedback: analysisResult.mental_boost || "Keep pushing forward!",

        // Player performance data
        performance: {
          hero: analysisResult.match_data?.hero || "Unknown",
          kda: formatKDA(analysisResult.match_data) || "N/A",
          damage: analysisResult.match_data?.hero_damage || "N/A",
          gpm: analysisResult.match_data?.gold || "N/A",
          match_duration: analysisResult.match_data?.match_duration || "N/A",
          rank: analysisResult.match_data?.rank || "Unknown",
        },

        // Quality metrics
        quality: {
          overall_confidence:
            (analysisResult.diagnostics?.confidence_score || 0) * 100,
          confidence_category: getConfidenceCategory(
            analysisResult.diagnostics?.confidence_score || 0
          ),
          warnings: analysisResult.diagnostics?.warnings || [],
          data_validation_passed:
            analysisResult.diagnostics?.hero_detected || false,
        },

        // Enhanced insights
        insights: {
          processing_time: 0,
          success_factors: ["Background processing completed successfully"],
          improvement_roadmap: extractImprovementPoints(
            analysisResult.feedback
          ),
          data_completeness: analysisResult.diagnostics?.data_completeness || 0,
        },
      },
      metadata: {
        job_id: jobId,
        analysis_mode: "async_worker",
        timestamp: new Date().toISOString(),
        backend_url: SKILLSHIFT_AI_BASE_URL,
      },
    };

    return NextResponse.json(dashboardResponse);
  } catch (error) {
    console.error("Job status API error:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to check job status",
        details: error.message,
        job_id: params.jobId,
      },
      { status: 500 }
    );
  }
}

function getProgressMessage(state) {
  switch (state) {
    case "queued":
      return "🔄 Analysis queued - waiting for worker...";
    case "started":
      return "🧠 AI analyzing your gameplay patterns...";
    case "deferred":
      return "⏳ Analysis scheduled - high demand detected...";
    default:
      return `📊 Processing... (${state})`;
  }
}

function getConfidenceCategory(score) {
  const percentage = score * 100;
  if (percentage >= 80) return "high";
  if (percentage >= 60) return "medium";
  return "low";
}

function formatKDA(matchData) {
  if (!matchData) return "N/A";

  const kills = matchData.kills || 0;
  const deaths = matchData.deaths || 0;
  const assists = matchData.assists || 0;

  return `${kills}/${deaths}/${assists}`;
}

function extractImprovementPoints(feedback) {
  if (!Array.isArray(feedback)) return [];

  return feedback
    .filter((item) => item.type === "warning" || item.type === "critical")
    .map((item) => item.message)
    .slice(0, 3); // Top 3 improvement points
}
