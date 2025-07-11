import { useState } from "react";
import FileUpload from "./components/FileUpload";
import ResultsDisplay from "./components/ResultsDisplay";

function App() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [useEnhanced, setUseEnhanced] = useState(true);
  const [playerIGN, setPlayerIGN] = useState("Lesz XVII");
  const [uploadedScreenshots, setUploadedScreenshots] = useState([]);
  const [analysisReady, setAnalysisReady] = useState(false);

  const handleFileUpload = async (file) => {
    // Add file to uploaded screenshots list
    const fileWithMetadata = {
      file: file,
      name: file.name,
      size: file.size,
      uploadedAt: new Date().toISOString(),
    };

    setUploadedScreenshots((prev) => {
      const newList = [...prev, fileWithMetadata];

      // Check if we can now analyze
      const canAnalyze = useEnhanced
        ? newList.length >= 2
        : newList.length >= 1;
      setAnalysisReady(canAnalyze);

      return newList;
    });
  };

  const handleRemoveFile = (index) => {
    setUploadedScreenshots((prev) => {
      const newList = prev.filter((_, i) => i !== index);

      // Update analysis readiness
      const canAnalyze = useEnhanced
        ? newList.length >= 2
        : newList.length >= 1;
      setAnalysisReady(canAnalyze);

      return newList;
    });
  };

  const handleAnalyze = async () => {
    if (!analysisReady || uploadedScreenshots.length === 0) {
      return;
    }

    setIsLoading(true);
    try {
      let analysisResults = [];

      // Process screenshots sequentially for session management
      let currentSessionId = sessionId; // Local variable to track session ID within loop
      
      for (let i = 0; i < uploadedScreenshots.length; i++) {
        const screenshot = uploadedScreenshots[i];
        const formData = new FormData();
        formData.append("file", screenshot.file);
        formData.append("ign", playerIGN);

        // Add session ID for subsequent uploads
        if (currentSessionId) {
          formData.append("session_id", currentSessionId);
        }

        const endpoint = useEnhanced ? "/api/analyze-enhanced" : "/api/analyze";
        const response = await fetch(endpoint, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Analysis failed for screenshot ${i + 1}`);
        }

        const result = await response.json();
        analysisResults.push(result);

        // Store session ID from first response for immediate use in next iteration
        if (i === 0 && result.session_info?.session_id) {
          currentSessionId = result.session_info.session_id;
          setSessionId(currentSessionId); // Also update React state
        }
      }

      // Use the most comprehensive result (usually the last one with full session data)
      const finalResult = analysisResults[analysisResults.length - 1];

      // Enhance with multi-screenshot context if available
      if (analysisResults.length > 1) {
        finalResult.multi_screenshot_analysis = {
          total_screenshots: analysisResults.length,
          confidence_improvement: true,
          screenshot_types: uploadedScreenshots.map((s) => s.name),
        };
      }

      setAnalysisResult(finalResult);
    } catch (error) {
      console.error("Error analyzing screenshots:", error);
      // Show user-friendly error
      setAnalysisResult({
        error: true,
        message: error.message || "Analysis failed. Please try again.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setAnalysisResult(null);
    setSessionId(null);
    setUploadedScreenshots([]);
    setAnalysisReady(false);
  };

  const handleNewSession = () => {
    setSessionId(null);
    setAnalysisResult(null);
    setUploadedScreenshots([]);
    setAnalysisReady(false);
  };

  const canAnalyze = useEnhanced
    ? uploadedScreenshots.length >= 2
    : uploadedScreenshots.length >= 1;

  return (
    <div className="min-h-screen bg-gradient-to-br from-mlbb-dark via-gray-900 to-mlbb-dark">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-mlbb-gold mb-2">
            SkillShift AI
          </h1>
          <p className="text-gray-300 text-lg">
            AI-powered coaching analysis for Mobile Legends: Bang Bang
          </p>

          {/* Enhanced Analysis Toggle */}
          <div className="mt-6 flex items-center justify-center space-x-4">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useEnhanced}
                onChange={(e) => {
                  setUseEnhanced(e.target.checked);
                  const canAnalyzeAfterToggle = e.target.checked
                    ? uploadedScreenshots.length >= 2
                    : uploadedScreenshots.length >= 1;
                  setAnalysisReady(canAnalyzeAfterToggle);
                }}
                className="w-4 h-4 text-mlbb-gold"
              />
              <span className="text-gray-300">
                📊 Enhanced Analysis (Multi-Screenshot)
              </span>
            </label>
          </div>

          {/* Player IGN Input */}
          <div className="mt-4 max-w-md mx-auto">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Player IGN
            </label>
            <input
              type="text"
              value={playerIGN}
              onChange={(e) => setPlayerIGN(e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-mlbb-gold focus:border-transparent"
              placeholder="Enter your in-game name"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <FileUpload
              onFileUpload={handleFileUpload}
              isLoading={isLoading}
              uploadedScreenshots={uploadedScreenshots}
              useEnhanced={useEnhanced}
              onRemoveFile={handleRemoveFile}
            />

            {/* Analysis Controls */}
            <div className="mt-6 space-y-3">
              <button
                onClick={handleAnalyze}
                disabled={!canAnalyze || isLoading}
                className={`w-full py-3 px-6 rounded-lg font-semibold transition-all ${
                  canAnalyze && !isLoading
                    ? "bg-mlbb-gold text-mlbb-dark hover:bg-yellow-400 shadow-lg"
                    : "bg-gray-600 text-gray-400 cursor-not-allowed"
                }`}
              >
                {isLoading
                  ? "Analyzing..."
                  : canAnalyze
                  ? `Analyze ${uploadedScreenshots.length} Screenshot${
                      uploadedScreenshots.length > 1 ? "s" : ""
                    }`
                  : useEnhanced
                  ? "Upload 2 screenshots to analyze"
                  : "Upload at least 1 screenshot"}
              </button>

              {uploadedScreenshots.length > 0 && (
                <button
                  onClick={handleReset}
                  className="w-full py-2 px-4 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors"
                >
                  Clear All Screenshots
                </button>
              )}
            </div>

            {/* Session Info */}
            {sessionId && (
              <div className="mt-4 p-3 bg-gray-700 rounded-lg">
                <p className="text-xs text-gray-400">
                  Session: {sessionId.substring(0, 8)}... (
                  {uploadedScreenshots.length} screenshots)
                </p>
                <button
                  onClick={handleNewSession}
                  className="text-sm text-mlbb-gold hover:text-yellow-400 transition-colors"
                >
                  New Session
                </button>
              </div>
            )}
          </div>

          <div>
            {analysisResult && (
              <ResultsDisplay
                result={analysisResult}
                onReset={handleReset}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
