import { NextResponse } from "next/server";
import FormData from "form-data";
import fetch from "node-fetch";

// Configuration for the skillshift-ai backend
const SKILLSHIFT_AI_BASE_URL =
  process.env.SKILLSHIFT_AI_URL || "http://localhost:8000";
const DEFAULT_IGN = process.env.DEFAULT_IGN || "Lesz XVII";
const USE_FAST_ANALYSIS = process.env.USE_FAST_ANALYSIS === "true";
const ANALYZE_ENDPOINT =
  process.env.NEXT_PUBLIC_ANALYZE_ENDPOINT || "/api/analyze";

// Experience stage configurations for enhanced feedback
const EXPERIENCE_STAGES = {
  first_upload: {
    anticipationMessage: "🧠 Decoding your first gameplay DNA...",
    revelationPrefix: "🎯 Your gameplay soul revealed:",
    focusAreas: ["mvp_detection", "basic_performance", "potential_discovery"],
  },
  mvp_revealed: {
    anticipationMessage: "🎮 Analyzing your tactical instincts...",
    revelationPrefix: "⚡ Your decision patterns unlocked:",
    focusAreas: ["decision_making", "positioning", "team_coordination"],
  },
  deeper_insights: {
    anticipationMessage: "📈 Mapping your growth trajectory...",
    revelationPrefix: "🚀 Your mastery evolution:",
    focusAreas: ["consistency", "improvement_trends", "advanced_tactics"],
  },
  mastery_journey: {
    anticipationMessage: "🏆 Revealing your competitive edge...",
    revelationPrefix: "⭐ Elite insights unlocked:",
    focusAreas: ["meta_adaptation", "strategic_depth", "leadership_potential"],
  },
};

export async function POST(request) {
  try {
    const formData = await request.formData();
    const file = formData.get("file");
    const ign = formData.get("ign") || DEFAULT_IGN;
    const analysisType = formData.get("analysisType") || "general";
    const experienceStage = formData.get("experienceStage") || "first_upload";

    if (!file) {
      return NextResponse.json(
        { success: false, error: "No file provided for analysis" },
        { status: 400 }
      );
    }

    // Validate file type - should be image
    if (!file.type.startsWith("image/")) {
      return NextResponse.json(
        {
          success: false,
          error: "Only image files are supported for analysis",
        },
        { status: 400 }
      );
    }

    // NEW: Health check before processing - use isolated endpoint
    try {
      const healthResponse = await fetch(
        `${SKILLSHIFT_AI_BASE_URL}/api/health-isolated`,
        {
          method: "GET",
          signal: AbortSignal.timeout(1000), // 1 second timeout for isolated health
        }
      );

      if (!healthResponse.ok) {
        console.error("Backend health check failed:", healthResponse.status);
        return NextResponse.json(
          {
            success: false,
            error: "Analysis service is currently unavailable",
            details: "Backend health check failed",
            stage: "health_check",
            backend_status: "degraded",
          },
          { status: 503 }
        );
      }
    } catch (healthError) {
      console.error("Backend connection failed:", healthError);
      return NextResponse.json(
        {
          success: false,
          error: "Cannot connect to analysis service",
          details: "Please ensure the backend is running on port 8000",
          stage: "connection",
          backend_status: "offline",
        },
        { status: 503 }
      );
    }

    // Get experience stage configuration
    const stageConfig =
      EXPERIENCE_STAGES[experienceStage] || EXPERIENCE_STAGES.first_upload;

    // Create FormData for skillshift-ai backend
    const backendFormData = new FormData();

    // Convert file to buffer for backend
    const fileBuffer = Buffer.from(await file.arrayBuffer());
    backendFormData.append("file", fileBuffer, {
      filename: file.name,
      contentType: file.type,
    });

    // Choose endpoint based on configuration - use env var or default to heavy analysis
    const endpoint = USE_FAST_ANALYSIS
      ? "/api/analyze-instant"
      : ANALYZE_ENDPOINT;
    const timeout = USE_FAST_ANALYSIS ? 5000 : 120000; // 5s for instant, 120s for heavy analysis

    // Call the skillshift-ai analysis endpoint with timeout
    let analysisResponse;
    try {
      analysisResponse = await fetch(
        `${SKILLSHIFT_AI_BASE_URL}${endpoint}?ign=${encodeURIComponent(ign)}`,
        {
          method: "POST",
          body: backendFormData,
          headers: backendFormData.getHeaders(),
          signal: AbortSignal.timeout(timeout),
        }
      );
    } catch (timeoutError) {
      console.log(
        "Analysis endpoint timed out, this should not happen with instant endpoint"
      );
      throw timeoutError; // Instant endpoint should never timeout
    }

    if (!analysisResponse.ok) {
      const errorText = await analysisResponse.text();
      console.error("Analysis backend error:", errorText);

      return NextResponse.json(
        {
          success: false,
          error: "Analysis failed",
          details: errorText,
          stage: "backend_analysis",
        },
        { status: analysisResponse.status }
      );
    }

    const analysisResult = await analysisResponse.json();

    // CRITICAL FIX: Validate backend response structure
    if (
      !analysisResult ||
      (!analysisResult.match_data && !analysisResult.parsed_data)
    ) {
      console.error("Invalid backend response structure:", analysisResult);
      return NextResponse.json(
        {
          success: false,
          error: "Invalid analysis response",
          details: "Backend returned malformed data",
          stage: "response_validation",
        },
        { status: 502 }
      );
    }

    // Enhanced analysis with experience-aware insights
    const enhancedInsights = generateExperienceAwareInsights(
      analysisResult,
      experienceStage,
      stageConfig
    );

    // CRITICAL FIX: Handle confidence score format inconsistency
    const rawConfidence =
      analysisResult.confidence_scores?.overall_confidence ||
      analysisResult.overall_confidence ||
      analysisResult.diagnostics?.confidence_score ||
      0;

    // Normalize confidence to 0-100 range (percentage)
    const normalizedConfidence =
      rawConfidence > 1 ? rawConfidence : rawConfidence * 100;

    // Validate confidence isn't inflated without data
    const hasValidData =
      analysisResult.parsed_data?.hero &&
      analysisResult.parsed_data?.hero !== "Unknown" &&
      analysisResult.parsed_data?.kda !== "N/A";

    const actualConfidence = hasValidData
      ? normalizedConfidence
      : Math.min(normalizedConfidence, 30);

    // Transform the response to match dashboard expectations
    const dashboardResponse = {
      success: true,
      message: "Analysis completed successfully",
      analysis: {
        // Core analysis results
        statistical_feedback: enhancedInsights.statistical_feedback,
        mental_feedback: enhancedInsights.mental_feedback,

        // Player performance data
        performance: {
          hero: analysisResult.parsed_data?.hero || "Unknown",
          kda: analysisResult.parsed_data?.kda || "N/A",
          damage: analysisResult.parsed_data?.damage || "N/A",
          gpm: analysisResult.parsed_data?.gpm || "N/A",
          match_duration: analysisResult.parsed_data?.match_duration || "N/A",
          rank: analysisResult.parsed_data?.rank || "Unknown",
        },

        // Confidence and quality metrics with validation
        quality: {
          overall_confidence: actualConfidence,
          confidence_category:
            actualConfidence >= 80
              ? "high"
              : actualConfidence >= 60
              ? "medium"
              : "low",
          component_scores:
            analysisResult.confidence_scores?.component_scores || {},
          warnings: analysisResult.parsing_warnings || [],
          data_validation_passed: hasValidData,
        },

        // Enhanced insights with experience progression
        insights: {
          processing_time:
            analysisResult.ultimate_analysis?.processing_time || 0,
          success_factors: enhancedInsights.success_factors || [],
          improvement_roadmap: enhancedInsights.improvement_roadmap || [],
          data_completeness:
            analysisResult.ultimate_analysis?.data_completeness || 0,
          experience_progression: enhancedInsights.experience_progression,
        },
      },
      experience: {
        stage: experienceStage,
        stage_config: stageConfig,
        progression_detected: enhancedInsights.progression_detected,
        unlocked_features: enhancedInsights.unlocked_features,
        next_invitation: enhancedInsights.next_invitation,
      },
      metadata: {
        ign: ign,
        analysisType: analysisType,
        filename: file.name,
        fileSize: file.size,
        timestamp: new Date().toISOString(),
        backend_url: SKILLSHIFT_AI_BASE_URL,
        experienceStage: experienceStage,
      },
    };

    return NextResponse.json(dashboardResponse);
  } catch (error) {
    console.error("Analysis API error:", error);

    return NextResponse.json(
      {
        success: false,
        error: "Analysis pipeline failed",
        details: error.message,
        stage: "api_processing",
      },
      { status: 500 }
    );
  }
}

function generateExperienceAwareInsights(
  analysisResult,
  experienceStage,
  stageConfig
) {
  const baseStatistical = analysisResult.statistical_feedback || [];
  const baseMental = analysisResult.mental_feedback || "";
  const baseSuccessFactors =
    analysisResult.ultimate_analysis?.success_factors || [];
  const baseRoadmap =
    analysisResult.ultimate_analysis?.improvement_roadmap || [];

  // Detect meaningful progressions based on stage
  const progressionDetected = detectProgression(
    analysisResult,
    experienceStage
  );
  const unlockedFeatures = determineUnlockedFeatures(
    analysisResult,
    experienceStage
  );
  const nextInvitation = generateNextInvitation(
    analysisResult,
    experienceStage
  );

  // Experience-aware enhancements
  let enhancedStatistical = [...baseStatistical];
  let enhancedMental = baseMental;
  let enhancedSuccessFactors = [...baseSuccessFactors];
  let enhancedRoadmap = [...baseRoadmap];

  switch (experienceStage) {
    case "first_upload":
      enhancedStatistical.unshift({
        message:
          "🎯 Welcome to your coaching journey! This analysis reveals your baseline gameplay DNA.",
        severity: "info",
        category: "experience_intro",
      });

      if (analysisResult.parsed_data?.rank?.includes("MVP")) {
        enhancedStatistical.push({
          message:
            "🏆 MVP performance detected! You have the potential for tactical leadership.",
          severity: "positive",
          category: "mvp_discovery",
        });
      }

      enhancedMental = `🧠 First Revelation: ${enhancedMental}\n\n🎯 This is just the beginning. Your gameplay patterns show ${getPersonalityHint(
        analysisResult
      )}. Ready to go deeper?`;
      break;

    case "mvp_revealed":
      enhancedStatistical.unshift({
        message:
          "⚡ Tactical Analysis: Your decision-making patterns are emerging from the data.",
        severity: "info",
        category: "tactical_focus",
      });

      enhancedMental = `🎮 Decision Pattern Analysis: ${enhancedMental}\n\n🧠 Your instincts are becoming visible. Each choice reveals your strategic DNA.`;
      break;

    case "deeper_insights":
      enhancedStatistical.unshift({
        message:
          "📈 Growth Trajectory: Analyzing consistency and improvement patterns across matches.",
        severity: "info",
        category: "growth_analysis",
      });

      enhancedMental = `🚀 Mastery Evolution: ${enhancedMental}\n\n📊 Your growth path is becoming clear. Patterns reveal your learning velocity.`;
      break;

    case "mastery_journey":
      enhancedStatistical.unshift({
        message:
          "⭐ Elite Analysis: Unlocking advanced strategic insights and meta adaptation patterns.",
        severity: "info",
        category: "elite_insights",
      });

      enhancedMental = `🏆 Competitive Edge: ${enhancedMental}\n\n⚡ You're operating at advanced levels. Time to explore leadership dynamics.`;
      break;
  }

  return {
    statistical_feedback: enhancedStatistical,
    mental_feedback: enhancedMental,
    success_factors: enhancedSuccessFactors,
    improvement_roadmap: enhancedRoadmap,
    progression_detected: progressionDetected,
    unlocked_features: unlockedFeatures,
    next_invitation: nextInvitation,
    experience_progression: {
      stage: experienceStage,
      insights_unlocked: enhancedStatistical.length,
      personality_hint: getPersonalityHint(analysisResult),
      mastery_indicators: getMasteryIndicators(analysisResult),
    },
  };
}

function detectProgression(analysisResult, experienceStage) {
  const confidence = analysisResult.confidence_scores?.overall_confidence || 0;
  const hasMVP = analysisResult.parsed_data?.rank?.includes("MVP");
  const hasGoodKDA =
    parseFloat(analysisResult.parsed_data?.kda?.split("/")[0] || "0") > 5;

  switch (experienceStage) {
    case "first_upload":
      return hasMVP || confidence > 0.8;
    case "mvp_revealed":
      return hasGoodKDA && confidence > 0.7;
    case "deeper_insights":
      return confidence > 0.85;
    default:
      return false;
  }
}

function determineUnlockedFeatures(analysisResult, experienceStage) {
  const features = [];
  const hasMVP = analysisResult.parsed_data?.rank?.includes("MVP");
  const confidence = analysisResult.confidence_scores?.overall_confidence || 0;

  if (experienceStage === "first_upload" && hasMVP) {
    features.push("video_analysis", "decision_tracking");
  }

  if (experienceStage === "mvp_revealed" && confidence > 0.7) {
    features.push("trend_analysis", "consistency_tracking");
  }

  if (experienceStage === "deeper_insights") {
    features.push("team_synergy", "meta_analysis");
  }

  return features;
}

function generateNextInvitation(analysisResult, experienceStage) {
  const hasMVP = analysisResult.parsed_data?.rank?.includes("MVP");
  const hero = analysisResult.parsed_data?.hero || "your hero";

  switch (experienceStage) {
    case "first_upload":
      if (hasMVP) {
        return {
          title: "🎮 Ready to see your instincts in action?",
          subtitle: `Your ${hero} shows MVP potential. Upload a gameplay video to see your real-time decision patterns.`,
          cta: "Analyze Decision-Making",
        };
      }
      return {
        title: "🧠 Want to understand your tactical mind?",
        subtitle:
          "Upload another match to reveal how your gameplay adapts under different conditions.",
        cta: "Continue Discovery",
      };

    case "mvp_revealed":
      return {
        title: "📈 Ready to map your growth journey?",
        subtitle:
          "Upload more matches to unlock your historical performance patterns and consistency metrics.",
        cta: "Unlock Growth Map",
      };

    case "deeper_insights":
      return {
        title: "⭐ Time for advanced mastery tools?",
        subtitle:
          "Access team synergy analysis, meta adaptation insights, and leadership potential assessment.",
        cta: "Explore Elite Features",
      };

    default:
      return {
        title: "🚀 Your journey continues...",
        subtitle: "Each upload reveals new layers of your competitive DNA.",
        cta: "Next Revelation",
      };
  }
}

function getPersonalityHint(analysisResult) {
  const kda = analysisResult.parsed_data?.kda || "0/0/0";
  const [kills, deaths, assists] = kda.split("/").map((x) => parseInt(x) || 0);

  if (kills > deaths * 2) {
    return "aggressive excellence - you strike decisively";
  } else if (assists > kills) {
    return "strategic support - you enable team victories";
  } else if (deaths < kills) {
    return "calculated aggression - you balance risk and reward";
  }
  return "adaptive potential - your style is emerging";
}

function getMasteryIndicators(analysisResult) {
  const indicators = [];
  const confidence = analysisResult.confidence_scores?.overall_confidence || 0;

  if (confidence > 0.9) indicators.push("elite_confidence");
  if (analysisResult.parsed_data?.rank?.includes("MVP"))
    indicators.push("mvp_performance");
  if (analysisResult.parsed_data?.damage) indicators.push("damage_dealing");

  return indicators;
}

export async function GET() {
  return NextResponse.json({
    message: "MLBB Coach AI Sequential Experience Analysis Endpoint",
    status: "operational",
    backend_url: SKILLSHIFT_AI_BASE_URL,
    supported_methods: ["POST"],
    required_fields: ["file"],
    optional_fields: ["ign", "analysisType", "experienceStage"],
    experience_stages: Object.keys(EXPERIENCE_STAGES),
  });
}
