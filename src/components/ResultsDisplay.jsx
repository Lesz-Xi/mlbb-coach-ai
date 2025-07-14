import React, { useState } from "react";

const ResultsDisplay = ({
  result,
  onReset,
  sessionInfo,
  debugInfo,
  warnings,
}) => {
  const [showDebugInfo, setShowDebugInfo] = useState(false);
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  // Check if we should show analysis results based on diagnostics
  const shouldShowAnalysis = () => {
    if (!result.diagnostics) return true; // Fallback for old responses

    const diag = result.diagnostics;

    // Required checks for reliable analysis
    const hasReliableData =
      diag.hero_detected === true &&
      diag.confidence_score >= 0.7 &&
      diag.gold_data_valid === true &&
      diag.kda_data_complete === true;

    // Enhanced mode has more lenient requirements if hero suggestions exist
    if (diag.analysis_mode === "enhanced" && !hasReliableData) {
      return (
        diag.confidence_score >= 0.6 &&
        (diag.hero_detected ||
          (diag.hero_suggestions && diag.hero_suggestions.length > 0)) &&
        diag.kda_data_complete
      );
    }

    return hasReliableData;
  };

  const getAnalysisQualityMessage = () => {
    if (!result.diagnostics) return [];

    const diag = result.diagnostics;
    const issues = [];

    if (!diag.hero_detected) {
      if (diag.hero_suggestions && diag.hero_suggestions.length > 0) {
        const topSuggestion = diag.hero_suggestions[0];
        issues.push(
          `Hero unclear - best guess: ${topSuggestion[0]} (${(
            topSuggestion[1] * 100
          ).toFixed(0)}%)`
        );
      } else {
        issues.push("Hero could not be identified");
      }
    }

    if (!diag.gold_data_valid) {
      issues.push("Gold/economy data missing or invalid");
    }

    if (!diag.kda_data_complete) {
      issues.push("KDA data incomplete");
    }

    if (diag.confidence_score < 0.7) {
      issues.push(
        `Low analysis confidence (${(diag.confidence_score * 100).toFixed(0)}%)`
      );
    }

    return issues;
  };

  const getSeverityColor = (type) => {
    switch (type) {
      case "critical":
        return "bg-red-900 border-red-500 text-red-100";
      case "warning":
        return "bg-yellow-900 border-yellow-500 text-yellow-100";
      case "info":
        return "bg-blue-900 border-blue-500 text-blue-100";
      default:
        return "bg-gray-900 border-gray-500 text-gray-100";
    }
  };

  const getSeverityIcon = (type) => {
    switch (type) {
      case "critical":
        return "🔴";
      case "warning":
        return "⚠️";
      case "info":
        return "💡";
      default:
        return "📝";
    }
  };

  const getRatingColor = (rating) => {
    if (!rating) return "text-gray-400";

    switch (rating.toLowerCase()) {
      case "excellent":
        return "text-green-400";
      case "good":
        return "text-blue-400";
      case "average":
        return "text-yellow-400";
      case "poor":
        return "text-red-400";
      default:
        return "text-gray-400";
    }
  };

  const getRatingBackgroundStyle = (rating) => {
    if (!rating) return "bg-gray-700";

    const lowPerformance = ["poor", "bronze", "needs improvement"].includes(
      rating.toLowerCase()
    );
    return lowPerformance ? "bg-gray-800 border-gray-600" : "bg-gray-700";
  };

  const getFeedbackContainerStyle = (rating) => {
    if (!rating) return "bg-gray-700";

    const lowPerformance = ["poor", "bronze", "needs improvement"].includes(
      rating.toLowerCase()
    );
    return lowPerformance
      ? "bg-gray-800 border border-gray-600"
      : "bg-gray-700";
  };

  const criticalFeedback =
    result.feedback?.filter((item) => item.type === "critical") || [];
  const warningFeedback =
    result.feedback?.filter((item) => item.type === "warning") || [];
  const infoFeedback =
    result.feedback?.filter((item) => item.type === "info") || [];

  const showAnalysis = shouldShowAnalysis();
  const qualityIssues = getAnalysisQualityMessage();

  return (
    <div className="bg-gray-800 rounded-lg p-8 shadow-xl">
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-2xl font-bold text-mlbb-gold">Analysis Results</h2>
        <button
          onClick={onReset}
          className="bg-mlbb-blue hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
        >
          Analyze New Screenshot
        </button>
      </div>

      {/* Analysis Quality Warning */}
      {!showAnalysis && (
        <div className="mb-8 p-6 bg-orange-900 border border-orange-500 rounded-lg">
          <div className="flex items-start space-x-3">
            <span className="text-2xl">⚠️</span>
            <div>
              <h3 className="text-lg font-semibold text-orange-200 mb-2">
                {result.diagnostics?.analysis_state === "partial"
                  ? "Partial Analysis Detected"
                  : "Analysis Quality Issues"}
              </h3>
              <p className="text-orange-100 mb-3">
                {result.diagnostics?.analysis_state === "partial"
                  ? `We detected some match data but confidence is low (${(
                      result.diagnostics.confidence_score * 100
                    ).toFixed(0)}%). Results may be incomplete.`
                  : "We weren't able to fully analyze your match. The following issues were detected:"}
              </p>
              <ul className="text-orange-200 text-sm space-y-1 mb-4">
                {qualityIssues?.map((issue, index) => (
                  <li key={index} className="flex items-start">
                    <span className="mr-2">•</span>
                    <span>{issue}</span>
                  </li>
                ))}
              </ul>

              {/* Show partial detection details if available */}
              {result.diagnostics?.analysis_state === "partial" && (
                <div className="bg-orange-800 border border-orange-700 rounded-lg p-3 mb-3">
                  <p className="text-orange-100 text-sm font-semibold mb-2">
                    📊 Partial Detection Results:
                  </p>
                  <ul className="text-orange-200 text-xs space-y-1">
                    {result.diagnostics.hero_confidence > 0 && (
                      <li>
                        • Hero: {result.diagnostics.hero_name} (
                        {(result.diagnostics.hero_confidence * 100).toFixed(0)}%
                        confidence)
                      </li>
                    )}
                    {result.diagnostics.kda_confidence > 0 && (
                      <li>
                        • KDA:{" "}
                        {(result.diagnostics.kda_confidence * 100).toFixed(0)}%
                        complete
                      </li>
                    )}
                    {result.diagnostics.gold_confidence > 0 && (
                      <li>
                        • Gold:{" "}
                        {(result.diagnostics.gold_confidence * 100).toFixed(0)}%
                        confidence
                      </li>
                    )}
                  </ul>
                </div>
              )}

              <div className="bg-orange-800 border border-orange-600 rounded-lg p-3">
                <p className="text-orange-100 text-sm font-semibold mb-2">
                  💡 Suggestions:
                </p>
                <ul className="text-orange-200 text-xs space-y-1">
                  <li>• Upload clearer, higher-resolution screenshots</li>
                  <li>• Enable Enhanced Analysis mode for better detection</li>
                  <li>
                    • Try uploading both scoreboard and match summary screens
                  </li>
                  <li>• Ensure hero portraits and text are clearly visible</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Show analysis only if quality is sufficient */}
      {showAnalysis && (
        <>
          {/* Overall Rating */}
          <div
            className={`mb-8 p-6 rounded-lg ${getRatingBackgroundStyle(
              result.overall_rating
            )}`}
          >
            <h3 className="text-lg font-semibold text-yellow-300 mb-2">
              Overall Performance
            </h3>
            <p
              className={`text-2xl font-bold ${getRatingColor(
                result.overall_rating
              )}`}
            >
              {result.overall_rating || "Unknown"}
            </p>
            {result.overall_rating &&
              ["poor", "bronze", "needs improvement"].includes(
                result.overall_rating.toLowerCase()
              ) && (
                <p className="text-gray-400 text-sm mt-2 italic">
                  Room for improvement in this match performance
                </p>
              )}
          </div>

          {/* Mental Boost */}
          <div
            className={`mb-8 p-6 rounded-lg ${
              result.overall_rating &&
              ["poor", "bronze", "needs improvement"].includes(
                result.overall_rating.toLowerCase()
              )
                ? "bg-gradient-to-r from-red-900 to-red-800 border border-red-500"
                : "bg-gradient-to-r from-mlbb-purple to-purple-600"
            }`}
          >
            <h3 className="text-lg font-semibold text-red-500 mb-2">
              Mental Boost
            </h3>
            <p className="text-white">{result.mental_boost}</p>
          </div>

          {/* Feedback Sections */}
          <div
            className={`space-y-6 p-6 rounded-lg ${getFeedbackContainerStyle(
              result.overall_rating
            )}`}
          >
            {criticalFeedback.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold text-red-400 mb-3">
                  Critical Issues ({criticalFeedback.length})
                </h3>
                <div className="space-y-3">
                  {criticalFeedback.map((item, index) => (
                    <div
                      key={index}
                      className={`p-4 rounded-lg border-l-4 ${getSeverityColor(
                        item.type
                      )}`}
                    >
                      <div className="flex items-start space-x-3">
                        <span className="text-lg">
                          {getSeverityIcon(item.type)}
                        </span>
                        <div>
                          <p className="font-semibold mb-1 text-white">
                            {item.category}
                          </p>
                          <p>{item.message}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {warningFeedback.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold text-red-400 mb-3">
                  Warnings ({warningFeedback.length})
                </h3>
                <div className="space-y-3">
                  {warningFeedback.map((item, index) => (
                    <div
                      key={index}
                      className={`p-4 rounded-lg border-l-4 ${getSeverityColor(
                        item.type
                      )}`}
                    >
                      <div className="flex items-start space-x-3">
                        <span className="text-lg">
                          {getSeverityIcon(item.type)}
                        </span>
                        <div>
                          <p className="font-semibold mb-1 text-white">
                            {item.category}
                          </p>
                          <p>{item.message}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {infoFeedback.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold text-blue-400 mb-3">
                  Tips & Insights ({infoFeedback.length})
                </h3>
                <div className="space-y-3">
                  {infoFeedback.map((item, index) => (
                    <div
                      key={index}
                      className={`p-4 rounded-lg border-l-4 ${getSeverityColor(
                        item.type
                      )}`}
                    >
                      <div className="flex items-start space-x-3">
                        <span className="text-lg">
                          {getSeverityIcon(item.type)}
                        </span>
                        <div>
                          <p className="font-semibold mb-1 text-white">
                            {item.category}
                          </p>
                          <p>{item.message}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {result.feedback && result.feedback.length === 0 && (
              <div className="text-center py-12">
                <p className="text-gray-400 text-lg">
                  No feedback items found.
                </p>
              </div>
            )}
          </div>
        </>
      )}

      {/* Diagnostics Panel */}
      {result.diagnostics && (
        <div className="mt-6">
          <button
            onClick={() => setShowDiagnostics(!showDiagnostics)}
            className="text-gray-400 hover:text-white text-sm flex items-center"
          >
            <span className="mr-2">{showDiagnostics ? "▼" : "▶"}</span>
            Diagnostics & Detection Info
          </button>

          {showDiagnostics && (
            <div className="mt-3 p-4 bg-gray-900 rounded-lg text-sm">
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <span className="text-gray-400">Analysis Mode:</span>
                  <span className="ml-2 text-white capitalize">
                    {result.diagnostics.analysis_mode}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Analysis State:</span>
                  <span
                    className={`ml-2 font-semibold capitalize ${
                      result.diagnostics.analysis_state === "complete"
                        ? "text-green-400"
                        : result.diagnostics.analysis_state === "partial"
                        ? "text-yellow-400"
                        : "text-red-400"
                    }`}
                  >
                    {result.diagnostics.analysis_state || "Unknown"}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Overall Confidence:</span>
                  <span
                    className={`ml-2 font-semibold ${
                      result.diagnostics.confidence_score >= 0.8
                        ? "text-green-400"
                        : result.diagnostics.confidence_score >= 0.6
                        ? "text-yellow-400"
                        : "text-red-400"
                    }`}
                  >
                    {(result.diagnostics.confidence_score * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Data Completeness:</span>
                  <span
                    className={`ml-2 ${
                      result.diagnostics.data_completeness >= 0.8
                        ? "text-green-400"
                        : result.diagnostics.data_completeness >= 0.6
                        ? "text-yellow-400"
                        : "text-red-400"
                    }`}
                  >
                    {(result.diagnostics.data_completeness * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Granular Confidence Scores */}
              <div className="border-t border-gray-700 pt-3 mb-3">
                <h4 className="text-xs text-gray-400 mb-2">
                  Detection Confidence:
                </h4>
                <div className="grid grid-cols-3 gap-4 text-xs">
                  <div className="flex flex-col">
                    <span className="text-gray-500 mb-1">Hero:</span>
                    <span
                      className={`font-medium ${
                        result.diagnostics.hero_confidence >= 0.7
                          ? "text-green-400"
                          : result.diagnostics.hero_confidence >= 0.3
                          ? "text-yellow-400"
                          : "text-red-400"
                      }`}
                    >
                      {result.diagnostics.hero_detected
                        ? `${result.diagnostics.hero_name} (${(
                            result.diagnostics.hero_confidence * 100
                          ).toFixed(0)}%)`
                        : `Not detected (${(
                            result.diagnostics.hero_confidence * 100
                          ).toFixed(0)}%)`}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-gray-500 mb-1">KDA:</span>
                    <span
                      className={`font-medium ${
                        result.diagnostics.kda_confidence >= 1.0
                          ? "text-green-400"
                          : result.diagnostics.kda_confidence >= 0.6
                          ? "text-yellow-400"
                          : "text-red-400"
                      }`}
                    >
                      {(result.diagnostics.kda_confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-gray-500 mb-1">Gold:</span>
                    <span
                      className={`font-medium ${
                        result.diagnostics.gold_confidence >= 0.8
                          ? "text-green-400"
                          : result.diagnostics.gold_confidence >= 0.5
                          ? "text-yellow-400"
                          : "text-red-400"
                      }`}
                    >
                      {(result.diagnostics.gold_confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>

              <div className="border-t border-gray-700 pt-3">
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div
                    className={`flex items-center ${
                      result.diagnostics.kda_data_complete
                        ? "text-green-400"
                        : "text-red-400"
                    }`}
                  >
                    <span className="mr-1">
                      {result.diagnostics.kda_data_complete ? "✓" : "✗"}
                    </span>
                    KDA Data Complete
                  </div>
                  <div
                    className={`flex items-center ${
                      result.diagnostics.gold_data_valid
                        ? "text-green-400"
                        : "text-red-400"
                    }`}
                  >
                    <span className="mr-1">
                      {result.diagnostics.gold_data_valid ? "✓" : "✗"}
                    </span>
                    Gold Data Valid
                  </div>
                  <div
                    className={`flex items-center ${
                      result.diagnostics.damage_data_available
                        ? "text-green-400"
                        : "text-yellow-400"
                    }`}
                  >
                    <span className="mr-1">
                      {result.diagnostics.damage_data_available ? "✓" : "~"}
                    </span>
                    Damage Data Available
                  </div>
                  <div
                    className={`flex items-center ${
                      result.diagnostics.match_duration_detected
                        ? "text-green-400"
                        : "text-yellow-400"
                    }`}
                  >
                    <span className="mr-1">
                      {result.diagnostics.match_duration_detected ? "✓" : "~"}
                    </span>
                    Match Duration Detected
                  </div>
                </div>

                {/* NEW: Extracted Data Values for Spatial Alignment Debugging */}
                {result.data && (
                  <div className="mt-3 pt-3 border-t border-gray-700">
                    <h4 className="text-xs text-gray-400 mb-2">
                      🔍 Extracted Data Values (Spatial Alignment Check):
                    </h4>
                    <div className="grid grid-cols-2 gap-4 text-xs">
                      <div className="space-y-1">
                        <div className="bg-gray-800 p-2 rounded">
                          <span className="text-yellow-400 font-medium">
                            KDA:
                          </span>
                          <span className="ml-2 text-white">
                            {result.data.kills || 0}/{result.data.deaths || 1}/
                            {result.data.assists || 0}
                          </span>
                        </div>
                        <div className="bg-gray-800 p-2 rounded">
                          <span className="text-yellow-400 font-medium">
                            Gold:
                          </span>
                          <span className="ml-2 text-white">
                            {result.data.gold
                              ? result.data.gold.toLocaleString()
                              : 0}
                          </span>
                        </div>
                      </div>
                      <div className="space-y-1">
                        <div className="bg-gray-800 p-2 rounded">
                          <span className="text-yellow-400 font-medium">
                            Hero Dmg:
                          </span>
                          <span className="ml-2 text-white">
                            {result.data.hero_damage
                              ? result.data.hero_damage.toLocaleString()
                              : 0}
                          </span>
                        </div>
                        <div className="bg-gray-800 p-2 rounded">
                          <span className="text-yellow-400 font-medium">
                            Dmg Taken:
                          </span>
                          <span className="ml-2 text-white">
                            {result.data.damage_taken
                              ? result.data.damage_taken.toLocaleString()
                              : 0}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="mt-2 text-xs text-gray-500 italic">
                      💡 Verify these values match your screenshot&apos;s player row
                      to confirm spatial alignment
                    </div>
                  </div>
                )}

                {result.diagnostics.hero_suggestions &&
                  result.diagnostics.hero_suggestions.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-700">
                      <span className="text-gray-400 text-xs">
                        Hero Suggestions:
                      </span>
                      <div className="mt-1">
                        {result.diagnostics.hero_suggestions.map(
                          (suggestion, index) => (
                            <span
                              key={index}
                              className="inline-block bg-gray-800 text-gray-300 px-2 py-1 rounded text-xs mr-2 mb-1"
                            >
                              {suggestion[0]} (
                              {(suggestion[1] * 100).toFixed(0)}%)
                            </span>
                          )
                        )}
                      </div>
                    </div>
                  )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Session Information */}
      {sessionInfo && (
        <div className="mt-8 p-6 bg-gray-300 rounded-lg">
          <h3 className="text-lg font-semibold text-white mb-4">
            Session Information
          </h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Session ID:</span>
              <span className="ml-2 text-white font-mono">
                {sessionInfo.session_id?.substring(0, 8)}...
              </span>
            </div>
            <div>
              <span className="text-gray-400">Screenshot Type:</span>
              <span className="ml-2 text-white capitalize">
                {sessionInfo.screenshot_type}
              </span>
            </div>
            <div>
              <span className="text-gray-400">Type Confidence:</span>
              <span className="ml-2 text-white">
                {(sessionInfo.type_confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              <span className="text-gray-400">Screenshots:</span>
              <span className="ml-2 text-white">
                {sessionInfo.screenshot_count}
              </span>
            </div>
            {sessionInfo.session_complete && (
              <div className="col-span-2">
                <span className="text-green-400">✓ Session Complete</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Warnings */}
      {warnings && warnings.length > 0 && (
        <div className="mt-6 p-4 bg-red-900 border-l-4 border-red-500 rounded-lg">
          <h4 className="text-red-300 font-semibold mb-2 flex items-center">
            <span className="text-lg mr-2">⚠️</span>
            Analysis Warnings - Reduced Accuracy
          </h4>
          <ul className="text-red-100 text-sm space-y-2">
            {warnings.map((warning, index) => (
              <li key={index} className="flex items-start">
                <span className="mr-2 text-red-400">⚠️</span>
                <span className="font-medium">{warning}</span>
              </li>
            ))}
          </ul>
          <p className="text-red-200 text-xs mt-3 italic">
            These warnings indicate potential issues with the analysis. Results
            may be less accurate.
          </p>
        </div>
      )}

      {/* Debug Information */}
      {debugInfo && Object.keys(debugInfo).length > 0 && (
        <div className="mt-6">
          <button
            onClick={() => setShowDebugInfo(!showDebugInfo)}
            className="text-gray-400 hover:text-white text-sm flex items-center"
          >
            <span className="mr-2">{showDebugInfo ? "▼" : "▶"}</span>
            Debug Information
          </button>

          {showDebugInfo && (
            <div className="mt-3 p-4 bg-gray-900 rounded-lg text-sm">
              {debugInfo.detected_keywords && (
                <div className="mb-3">
                  <span className="text-gray-400">Detected Keywords:</span>
                  <span className="ml-2 text-white">
                    {debugInfo.detected_keywords.join(", ")}
                  </span>
                </div>
              )}

              {debugInfo.hero_suggestions &&
                debugInfo.hero_suggestions.length > 0 && (
                  <div className="mb-3">
                    <span className="text-gray-400">Hero Suggestions:</span>
                    <ul className="ml-4 mt-1">
                      {debugInfo.hero_suggestions
                        .slice(0, 3)
                        .map((suggestion, index) => (
                          <li key={index} className="text-white">
                            {suggestion[0]} ({(suggestion[1] * 100).toFixed(1)}
                            %)
                          </li>
                        ))}
                    </ul>
                  </div>
                )}

              {debugInfo.hero_debug && (
                <div className="mb-3">
                  <span className="text-gray-400">Hero Detection:</span>
                  <ul className="ml-4 mt-1 text-sm">
                    <li className="text-white">
                      Strategies:{" "}
                      {debugInfo.hero_debug.strategies_tried?.join(", ") ||
                        "None"}
                    </li>
                    {debugInfo.hero_debug.manual_override && (
                      <li className="text-green-400">Manual override used</li>
                    )}
                    {debugInfo.hero_debug.error && (
                      <li className="text-red-400">
                        Error: {debugInfo.hero_debug.error}
                      </li>
                    )}
                  </ul>
                </div>
              )}

              <div className="text-xs text-gray-500 mt-3">
                <pre className="whitespace-pre-wrap">
                  {JSON.stringify(debugInfo, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ResultsDisplay;
