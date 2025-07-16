"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import ClientOnly from "@/components/ClientOnly";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import {
  Upload,
  Image,
  FileText,
  Trash2,
  CheckCircle,
  Clock,
  AlertCircle,
  Eye,
  Download,
  X,
  Brain,
  Target,
  Zap,
  Camera,
  Award,
  Loader2,
  Activity,
  TrendingUp,
  RefreshCw,
  WifiOff,
} from "lucide-react";

export default function ScreenshotAnalysisPage() {
  // FIXED: Removed all phantom mock data - clean slate
  const [uploadedFiles, setUploadedFiles] = useState([]);

  const [selectedFile, setSelectedFile] = useState(null);
  const [processingFiles, setProcessingFiles] = useState(new Set());
  const [analysisResults, setAnalysisResults] = useState({});
  const [selectedResultFile, setSelectedResultFile] = useState(null);
  const [showResultModal, setShowResultModal] = useState(false);

  // NEW: Streamlined analysis state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [completedResults, setCompletedResults] = useState(null);
  const [isViewingResults, setIsViewingResults] = useState(false);

  // IGN (In-Game Name) state - properly handle hydration
  const [playerIGN, setPlayerIGN] = useState("");

  // Load from localStorage only on client side
  useEffect(() => {
    const savedIGN = localStorage.getItem("mlbb_player_ign") || "";
    setPlayerIGN(savedIGN);
    // Initialize sync time only on client side
    setLastSyncTime(Date.now());
  }, []);

  // NEW: Backend sync and timeout management
  const [backendStatus, setBackendStatus] = useState("healthy");
  const [lastSyncTime, setLastSyncTime] = useState(null);
  const timeoutRefs = useRef(new Map());
  const statusCheckInterval = useRef(null);

  // NEW: Timeout configuration - increased for heavy analysis
  const PROCESSING_TIMEOUT = 120000; // 120 seconds
  const STATUS_CHECK_INTERVAL = 10000; // 10 seconds
  const MAX_RETRIES = 2;

  // NEW: Retry and toast state
  const [retryAttempts, setRetryAttempts] = useState(new Map());
  const [toastMessage, setToastMessage] = useState(null);
  const [healthCheckInterval, setHealthCheckInterval] = useState(3000); // Start at 3s

  // NEW: Backend status synchronization
  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        // Try both health endpoints for maximum compatibility
        let response;
        try {
          response = await fetch("/api/health", { method: "GET" });
        } catch (apiError) {
          console.warn(
            "Primary health endpoint failed, trying fallback:",
            apiError
          );
          response = await fetch("/health", { method: "GET" });
        }

        if (response.ok) {
          const healthData = await response.json();
          console.log("✅ Backend health check successful:", healthData);
          setBackendStatus("healthy");
          setLastSyncTime(Date.now());
          // Reset interval to 3s on success
          setHealthCheckInterval(3000);
        } else {
          console.warn(
            "Backend health check returned non-200:",
            response.status
          );
          setBackendStatus("degraded");
          // Double interval on failure
          setHealthCheckInterval((prev) => Math.min(prev * 2, 12000)); // Cap at 12s
        }
      } catch (error) {
        console.warn("Backend health check failed:", error);
        setBackendStatus("offline");
        // Double interval on failure
        setHealthCheckInterval((prev) => Math.min(prev * 2, 12000)); // Cap at 12s
      }
    };

    // Initial check
    checkBackendStatus();

    // Set up periodic health checks with adaptive interval
    const scheduleNextCheck = () => {
      if (statusCheckInterval.current) {
        clearTimeout(statusCheckInterval.current);
      }
      statusCheckInterval.current = setTimeout(() => {
        checkBackendStatus();
        scheduleNextCheck();
      }, healthCheckInterval);
    };

    scheduleNextCheck();

    return () => {
      if (statusCheckInterval.current) {
        clearTimeout(statusCheckInterval.current);
      }
    };
  }, [healthCheckInterval]);

  // NEW: Toast display component
  const ToastDisplay = () => {
    if (!toastMessage) return null;

    return (
      <div className="fixed top-4 right-4 z-50 bg-red-900/90 border border-red-500/50 rounded-lg p-4 max-w-md animate-in slide-in-from-right-4 fade-in-0 duration-300">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
          <div>
            <h4 className="text-red-400 font-medium">Analysis Failed</h4>
            <p className="text-gray-300 text-sm">{toastMessage}</p>
          </div>
          <button
            onClick={() => setToastMessage(null)}
            className="text-gray-400 hover:text-white ml-2"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  };

  // NEW: Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      timeoutRefs.current.forEach((timeout) => clearTimeout(timeout));
      timeoutRefs.current.clear();
    };
  }, []);

  // Save IGN to localStorage when it changes
  useEffect(() => {
    if (playerIGN) {
      localStorage.setItem("mlbb_player_ign", playerIGN);
    }
  }, [playerIGN]);

  // NEW: Processing timeout handler
  const handleProcessingTimeout = useCallback((fileId) => {
    console.warn(`Processing timeout for file ${fileId}`);

    setUploadedFiles((prev) =>
      prev.map((file) =>
        file.id === fileId
          ? {
              ...file,
              status: "error",
              results: "Analysis timed out - please retry",
            }
          : file
      )
    );

    // Remove from processing set
    setProcessingFiles((prev) => {
      const newSet = new Set(prev);
      newSet.delete(fileId);
      return newSet;
    });

    // Clear the timeout reference
    if (timeoutRefs.current.has(fileId)) {
      clearTimeout(timeoutRefs.current.get(fileId));
      timeoutRefs.current.delete(fileId);
    }
  }, []);

  // NEW: Retry failed analysis
  const handleRetryAnalysis = useCallback((fileId) => {
    setUploadedFiles((prev) =>
      prev.map((file) =>
        file.id === fileId
          ? { ...file, status: "queued", results: "Queued for retry..." }
          : file
      )
    );
  }, []);

  // NEW: Manual backend reconnection
  const handleReconnectBackend = useCallback(async () => {
    console.log("🔌 Attempting manual backend reconnection...");
    setBackendStatus("connecting");

    try {
      // Try multiple endpoints
      const endpoints = ["/api/health", "/health", "/"];
      let connected = false;

      for (const endpoint of endpoints) {
        try {
          const response = await fetch(endpoint, { method: "GET" });
          if (response.ok) {
            console.log(`✅ Connected via ${endpoint}`);
            setBackendStatus("healthy");
            setLastSyncTime(Date.now());
            connected = true;
            break;
          }
        } catch (error) {
          console.warn(`Failed to connect via ${endpoint}:`, error);
        }
      }

      if (!connected) {
        setBackendStatus("offline");
        console.error("❌ All connection attempts failed");
      }
    } catch (error) {
      console.error("❌ Manual reconnection failed:", error);
      setBackendStatus("offline");
    }
  }, []);

  const handleFileUpload = useCallback(
    (event) => {
      const files = Array.from(event.target.files);

      // Prevent multiple simultaneous uploads
      if (processingFiles.size > 0) {
        console.warn(
          "Analysis in progress, please wait before uploading more files"
        );
        return;
      }

      const validFiles = files.filter((file) => {
        if (!file.type.startsWith("image/")) {
          console.warn(`Skipping non-image file: ${file.name}`);
          return false;
        }
        return true;
      });

      if (validFiles.length === 0) {
        alert("Only image files are supported in Screenshot Analysis");
        return;
      }

      const newFiles = validFiles.map((file) => {
        // Generate unique ID with timestamp to prevent collisions
        const uniqueId = `${Date.now()}_${Math.random()
          .toString(36)
          .substr(2, 9)}`;

        return {
          id: uniqueId,
          name: file.name,
          type: "image",
          size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
          status: "queued",
          uploadDate: new Date().toLocaleString(),
          analysisType: "Screenshot Analysis",
          confidence: null,
          results: "Queued for analysis...",
          file: file,
          retryCount: 0,
          startTime: null, // Will be set when analysis starts
        };
      });

      console.log(`🔄 ${newFiles.length} file(s) uploaded successfully`);
      setUploadedFiles((prev) => [...prev, ...newFiles]);

      // Reset analysis state when new files are uploaded
      setIsAnalyzing(false);
      setAnalysisComplete(false);
      setCompletedResults(null);
      setIsViewingResults(false);

      // Clear the input to allow re-uploading same file
      event.target.value = "";
    },
    [processingFiles.size]
  );

  const handleAnalyze = useCallback(
    async (fileId) => {
      console.log(`🎯 Starting analysis for file ID: ${fileId}`);

      // Prevent duplicate processing
      if (processingFiles.has(fileId)) {
        console.warn(`File ${fileId} is already being processed`);
        return;
      }

      // Add to processing set
      setProcessingFiles((prev) => new Set([...prev, fileId]));

      // Update file status to processing and set start time
      const startTime = Date.now();
      setUploadedFiles((prev) =>
        prev.map((file) =>
          file.id === fileId
            ? {
                ...file,
                status: "processing",
                results: "Agent is analyzing...",
                startTime: startTime,
              }
            : file
        )
      );

      // NEW: Set up timeout for this file
      const timeoutId = setTimeout(() => {
        handleProcessingTimeout(fileId);
      }, PROCESSING_TIMEOUT);
      timeoutRefs.current.set(fileId, timeoutId);

      const attemptAnalysis = async (attemptNumber = 1) => {
        try {
          const file = uploadedFiles.find((f) => f.id === fileId);
          if (!file || !file.file) {
            throw new Error("File not found or invalid");
          }

          console.log(
            `📡 Sending request to backend for: ${file.name} (attempt ${attemptNumber})`
          );

          const formData = new FormData();
          formData.append("file", file.file);
          formData.append("ign", playerIGN || "Unknown Player");

          let response;
          try {
            // Add timeout to fetch request
            const controller = new AbortController();
            const timeoutId = setTimeout(
              () => controller.abort(),
              PROCESSING_TIMEOUT
            );

            response = await fetch("/api/analyze", {
              method: "POST",
              body: formData,
              signal: controller.signal,
            });

            clearTimeout(timeoutId);
          } catch (fetchError) {
            // Handle different types of fetch errors
            if (fetchError.name === "AbortError") {
              throw new Error("Request timed out - please try again");
            }
            // If /api/analyze fails, backend might be on different endpoint
            console.warn(
              "Primary analyze endpoint failed, checking backend status..."
            );
            await handleReconnectBackend();
            throw new Error(
              "Backend connection lost - please try again after reconnection"
            );
          }

          // NEW: Graceful retry logic for 429/503
          if (response.status === 429 || response.status === 503) {
            if (attemptNumber < 5) {
              // Up to 5 retries
              console.log(
                `⏳ Server busy (${response.status}), retrying in 7s... (attempt ${attemptNumber}/5)`
              );

              // Update file status to show retry
              setUploadedFiles((prev) =>
                prev.map((f) =>
                  f.id === fileId
                    ? {
                        ...f,
                        results: `Server busy, retrying in 7s... (${attemptNumber}/5)`,
                      }
                    : f
                )
              );

              // Wait 7 seconds before retry
              await new Promise((resolve) => setTimeout(resolve, 7000));

              // Update status to show retry in progress
              setUploadedFiles((prev) =>
                prev.map((f) =>
                  f.id === fileId
                    ? {
                        ...f,
                        results: `Retrying analysis... (attempt ${
                          attemptNumber + 1
                        }/5)`,
                      }
                    : f
                )
              );

              return attemptAnalysis(attemptNumber + 1);
            } else {
              // All retries failed
              setToastMessage("Server is busy, try again in a minute.");
              throw new Error(
                "Server is busy after 5 retry attempts. Please try again in a minute."
              );
            }
          }

          if (!response.ok) {
            // Check if it's a connection issue
            if (response.status >= 500) {
              setBackendStatus("degraded");
            }

            // CRITICAL FIX: Check for specific backend unavailability
            if (response.status === 503) {
              const errorData = await response.json();
              console.error("Backend unavailable:", errorData);
              throw new Error(
                errorData.error || "Analysis service temporarily unavailable"
              );
            }

            throw new Error(
              `Backend returned ${response.status}: ${response.statusText}`
            );
          }

          const result = await response.json();

          // CRITICAL FIX: Validate result has actual data
          // Handle both new API format (result.analysis) and legacy format (direct result)
          const analysisData = result.analysis || result;
          const hasValidData =
            analysisData &&
            (analysisData.statistical_feedback ||
              analysisData.parsed_data ||
              analysisData.match_data ||
              analysisData.mental_feedback);

          if (!hasValidData) {
            console.error("Invalid analysis result:", result);
            throw new Error(
              result.error || "Analysis failed - no data extracted"
            );
          }

          // CRITICAL FIX: Check for inflated confidence with no data
          const quality =
            analysisData?.confidence_scores || analysisData?.quality || {};
          const parsedData =
            analysisData?.parsed_data || analysisData?.match_data || {};

          if (
            quality.overall_confidence > 50 &&
            parsedData.hero === "Unknown" &&
            parsedData.kda === "N/A"
          ) {
            console.warn(
              "Detected inflated confidence with no data extraction"
            );
            quality.overall_confidence = 0;
            quality.confidence_category = "failed";
            if (analysisData.confidence_scores) {
              analysisData.confidence_scores = quality;
            }
          }
          console.log(`✅ Analysis complete for: ${file.name}`, result);

          // Clear timeout since processing completed successfully
          if (timeoutRefs.current.has(fileId)) {
            clearTimeout(timeoutRefs.current.get(fileId));
            timeoutRefs.current.delete(fileId);
          }

          // Update file with results
          setUploadedFiles((prev) =>
            prev.map((f) =>
              f.id === fileId
                ? {
                    ...f,
                    status: "analyzed",
                    confidence: Math.round(result.overall_confidence || 85),
                    results: `${
                      result.parsed_data?.hero || "Unknown Hero"
                    } - Analysis Complete`,
                  }
                : f
            )
          );

          // Store detailed results
          setAnalysisResults((prev) => ({
            ...prev,
            [fileId]: result,
          }));

          // NEW: Set completion state for streamlined UI
          const processingTime = Date.now() - (file.startTime || Date.now());
          const confidence =
            quality.overall_confidence ||
            result.confidence_scores?.overall_confidence ||
            85;

          setIsAnalyzing(false);
          setAnalysisComplete(true);
          setCompletedResults({
            ...result,
            processingTime: processingTime / 1000, // Convert to seconds
            confidence: Math.round(confidence),
            fileName: file.name,
          });

          console.log(
            `🎯 Analysis UI state updated: complete=true, confidence=${confidence}%`
          );
        } catch (error) {
          console.error(`❌ Analysis failed for file ${fileId}:`, error);

          // Reset analysis state on error
          setIsAnalyzing(false);
          setAnalysisComplete(false);
          setCompletedResults(null);
          setIsViewingResults(false);

          // Clear timeout
          if (timeoutRefs.current.has(fileId)) {
            clearTimeout(timeoutRefs.current.get(fileId));
            timeoutRefs.current.delete(fileId);
          }

          // Increment retry count
          const currentFile = uploadedFiles.find((f) => f.id === fileId);
          const retryCount = (currentFile?.retryCount || 0) + 1;

          // Update file with error status
          setUploadedFiles((prev) =>
            prev.map((f) =>
              f.id === fileId
                ? {
                    ...f,
                    status: "error",
                    results: `Analysis failed: ${error.message}${
                      retryCount < MAX_RETRIES ? " (retry available)" : ""
                    }`,
                    retryCount: retryCount,
                  }
                : f
            )
          );

          throw error; // Re-throw to prevent multiple calls
        }
      };

      try {
        await attemptAnalysis();
      } catch (error) {
        // Final error handling - this is reached only if all retries fail
        console.error(`🚫 Final analysis failure for file ${fileId}:`, error);
      } finally {
        // Always remove from processing set
        setProcessingFiles((prev) => {
          const newSet = new Set(prev);
          newSet.delete(fileId);
          return newSet;
        });
      }
    },
    [uploadedFiles, handleProcessingTimeout, playerIGN, handleReconnectBackend]
  );

  // NEW: Handle streamlined results viewing
  const handleViewCompletedResults = useCallback(() => {
    setIsViewingResults(true);

    if (completedResults && uploadedFiles.length > 0) {
      const completedFile = uploadedFiles.find((f) => f.status === "analyzed");
      if (completedFile) {
        setSelectedResultFile(completedFile);
        setShowResultModal(true);
      }
    } else if (completedResults) {
      // If we have results but no file reference, create a mock file for the modal
      const mockFile = {
        name: completedResults.fileName || "Analysis Results",
        id: "completed_analysis",
        status: "analyzed",
      };
      setSelectedResultFile(mockFile);
      setShowResultModal(true);
    }
  }, [completedResults, uploadedFiles]);

  const handleAnalyzeAll = useCallback(async () => {
    const queuedFiles = uploadedFiles.filter((f) => f.status === "queued");

    if (queuedFiles.length === 0) {
      console.warn("No files queued for analysis");
      return;
    }

    if (processingFiles.size > 0) {
      console.warn("Analysis already in progress, please wait");
      return;
    }

    console.log(`🚀 Batch analysis started for ${queuedFiles.length} files`);

    // NEW: Set streamlined analysis state
    setIsAnalyzing(true);
    setAnalysisComplete(false);
    setCompletedResults(null);

    console.log(`🚀 Analysis started: analyzing=${true}, complete=${false}`);

    // Process files sequentially to prevent browser hanging
    for (let i = 0; i < queuedFiles.length; i++) {
      const file = queuedFiles[i];
      try {
        await handleAnalyze(file.id);
        // Longer delay between requests to prevent overwhelming the browser and backend
        await new Promise((resolve) => setTimeout(resolve, 500));

        // Add progress indicator
        console.log(
          `✅ Progress: ${i + 1}/${queuedFiles.length} files processed`
        );

        // Allow browser to breathe by yielding control
        await new Promise((resolve) => requestAnimationFrame(resolve));
      } catch (error) {
        console.error(`Failed to analyze file ${file.id}:`, error);
        // Continue with next file even if one fails
      }
    }
  }, [uploadedFiles, handleAnalyze, processingFiles.size]);

  // NEW: Force refresh processing status
  const handleRefreshStatus = useCallback(() => {
    console.log("🔄 Refreshing analysis status...");

    // Reset any stuck processing files
    setProcessingFiles(new Set());

    // Clear all timeouts
    timeoutRefs.current.forEach((timeout) => clearTimeout(timeout));
    timeoutRefs.current.clear();

    // Reset stuck processing files to queued
    setUploadedFiles((prev) =>
      prev.map((file) =>
        file.status === "processing"
          ? { ...file, status: "queued", results: "Reset - ready for analysis" }
          : file
      )
    );
  }, []);



  const handleDelete = useCallback((fileId) => {
    console.log(`🗑️ Deleting file ID: ${fileId}`);

    // Clear any associated timeout
    if (timeoutRefs.current.has(fileId)) {
      clearTimeout(timeoutRefs.current.get(fileId));
      timeoutRefs.current.delete(fileId);
    }

    // Remove from processing set
    setProcessingFiles((prev) => {
      const newSet = new Set(prev);
      newSet.delete(fileId);
      return newSet;
    });

    // Remove file and results
    setUploadedFiles((prev) => {
      const filteredFiles = prev.filter((file) => file.id !== fileId);

      // Reset analysis state if no files left
      if (filteredFiles.length === 0) {
        setIsAnalyzing(false);
        setAnalysisComplete(false);
        setCompletedResults(null);
        setIsViewingResults(false);
      }

      return filteredFiles;
    });

    setAnalysisResults((prev) => {
      const newResults = { ...prev };
      delete newResults[fileId];
      return newResults;
    });
  }, []);

  const handleViewResults = useCallback((file) => {
    setSelectedResultFile(file);
    setShowResultModal(true);
  }, []);

  const getStatusIcon = (status, fileId) => {
    switch (status) {
      case "analyzed":
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case "processing":
        return (
          <div className="flex items-center gap-1">
            <Loader2 className="w-4 h-4 text-orange-500 animate-spin" />
            <Activity className="w-3 h-3 text-orange-400 animate-pulse" />
          </div>
        );
      case "error":
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-blue-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "analyzed":
        return "border-green-500 text-green-500";
      case "processing":
        return "border-orange-500 text-orange-500";
      case "error":
        return "border-red-500 text-red-500";
      default:
        return "border-blue-500 text-blue-500";
    }
  };

  const getProcessingMessage = (status) => {
    if (status === "processing") {
      const messages = [
        "Agent is analyzing...",
        "Detecting heroes and medals...",
        "Extracting performance metrics...",
        "Calculating confidence scores...",
        "Generating insights...",
      ];
      // Only use random on client side to prevent hydration issues
      if (typeof window !== "undefined") {
        return messages[Math.floor(Math.random() * messages.length)];
      }
      // Fallback for SSR
      return messages[0];
    }
    return null;
  };

  // Enhanced Results Modal Component
  const ResultsModal = () => {
    if (!selectedResultFile || !showResultModal) return null;

    const results = analysisResults[selectedResultFile.id] || completedResults;

    return (
      <Dialog
        open={showResultModal}
        onOpenChange={(open) => {
          setShowResultModal(open);
          if (!open) {
            setIsViewingResults(false);
          }
        }}
      >
        <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto bg-gray-900 border-2 border-orange-500/30 shadow-2xl animate-in fade-in-0 zoom-in-95 duration-300">
          <div className="absolute inset-0 bg-gradient-to-br from-orange-900/20 to-yellow-900/20 pointer-events-none" />

          <DialogHeader className="relative z-10">
            <DialogTitle className="text-2xl font-bold bg-gradient-to-r from-orange-400 to-yellow-400 bg-clip-text text-transparent">
              🎯 Analysis Results: {selectedResultFile.name}
            </DialogTitle>
          </DialogHeader>

          <div className="relative z-10 mt-6 space-y-6">
            {results ? (
              <>
                {/* Performance Overview */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card className="bg-orange-950/30 border-orange-500/20">
                    <CardHeader>
                      <CardTitle className="text-orange-400 flex items-center gap-2">
                        <Target className="w-5 h-5" />
                        Performance Metrics
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="text-center p-3 bg-gray-800/50 rounded-lg">
                          <div className="text-2xl font-bold text-white">
                            {results.parsed_data?.hero || "Unknown"}
                          </div>
                          <div className="text-sm text-gray-400">Hero</div>
                        </div>
                        <div className="text-center p-3 bg-gray-800/50 rounded-lg">
                          <div className="text-2xl font-bold text-green-400">
                            {results.parsed_data?.kda || "N/A"}
                          </div>
                          <div className="text-sm text-gray-400">KDA</div>
                        </div>
                        <div className="text-center p-3 bg-gray-800/50 rounded-lg">
                          <div className="text-2xl font-bold text-orange-400">
                            {results.parsed_data?.damage || "N/A"}
                          </div>
                          <div className="text-sm text-gray-400">Damage</div>
                        </div>
                        <div className="text-center p-3 bg-gray-800/50 rounded-lg">
                          <div className="text-2xl font-bold text-yellow-400">
                            {results.parsed_data?.gold || "N/A"}
                          </div>
                          <div className="text-sm text-gray-400">Gold</div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-yellow-950/30 border-yellow-500/20">
                    <CardHeader>
                      <CardTitle className="text-yellow-400 flex items-center gap-2">
                        <TrendingUp className="w-5 h-5" />
                        Analysis Quality
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-3">
                        <div>
                          <div className="flex justify-between mb-1">
                            <span className="text-gray-400">
                              Overall Confidence
                            </span>
                            <span className="text-white font-bold">
                              {Math.round(results.overall_confidence || 0)}%
                            </span>
                          </div>
                          <Progress
                            value={results.overall_confidence || 0}
                            className="h-2"
                          />
                        </div>
                        <div>
                          <div className="flex justify-between mb-1">
                            <span className="text-gray-400">
                              Data Completeness
                            </span>
                            <span className="text-white font-bold">
                              {Math.round(
                                (results.completeness_score || 0) * 100
                              )}
                              %
                            </span>
                          </div>
                          <Progress
                            value={(results.completeness_score || 0) * 100}
                            className="h-2"
                          />
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Coaching Insights */}
                {results.coaching_insights && (
                  <Card className="bg-orange-950/30 border-orange-500/20">
                    <CardHeader>
                      <CardTitle className="text-orange-400 flex items-center gap-2">
                        <Brain className="w-5 h-5" />
                        AI Coaching Insights
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <h4 className="text-green-400 font-medium mb-3 flex items-center gap-2">
                          <CheckCircle className="w-4 h-4" />
                          Strengths
                        </h4>
                        <ul className="space-y-2">
                          {(results.coaching_insights.strengths || []).map(
                            (strength, index) => (
                              <li key={index} className="text-gray-300 text-sm">
                                • {strength}
                              </li>
                            )
                          )}
                        </ul>
                      </div>
                      <div>
                        <h4 className="text-yellow-400 font-medium mb-3 flex items-center gap-2">
                          <AlertCircle className="w-4 h-4" />
                          Areas to Improve
                        </h4>
                        <ul className="space-y-2">
                          {(results.coaching_insights.weaknesses || []).map(
                            (weakness, index) => (
                              <li key={index} className="text-gray-300 text-sm">
                                • {weakness}
                              </li>
                            )
                          )}
                        </ul>
                      </div>
                      <div>
                        <h4 className="text-blue-400 font-medium mb-3 flex items-center gap-2">
                          <Target className="w-4 h-4" />
                          Recommendations
                        </h4>
                        <ul className="space-y-2">
                          {(
                            results.coaching_insights.recommendations || []
                          ).map((rec, index) => (
                            <li key={index} className="text-gray-300 text-sm">
                              • {rec}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </>
            ) : (
              <div className="text-center py-12">
                <Brain className="w-16 h-16 text-gray-500 mx-auto mb-4 animate-pulse" />
                <p className="text-gray-400 text-lg">
                  No detailed analysis results available
                </p>
                <p className="text-gray-500 text-sm mt-2">
                  Try re-analyzing this screenshot
                </p>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    );
  };

  const processingCount = processingFiles.size;
  const queuedCount = uploadedFiles.filter((f) => f.status === "queued").length;
  const errorCount = uploadedFiles.filter((f) => f.status === "error").length;

  return (
    <div className="p-6 space-y-6">
      {/* NEW: Toast notifications */}
      <ToastDisplay />

      {/* Header with Backend Status */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-orange-400 to-yellow-400 bg-clip-text text-transparent">
            Screenshot Analysis
          </h1>
          <p className="text-gray-300 mt-2">
            Upload MLBB screenshots for instant performance analysis • KDA
            Detection • Medal Recognition • Hero Identification
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Backend Status Indicator */}
          <Badge
            variant="outline"
            className={
              backendStatus === "healthy"
                ? "border-green-500 text-green-500"
                : backendStatus === "degraded"
                ? "border-yellow-500 text-yellow-500"
                : "border-red-500 text-red-500"
            }
          >
            {backendStatus === "healthy" && "🟢"}
            {backendStatus === "degraded" && "🟡"}
            {backendStatus === "offline" && "🔴"}
            Backend {backendStatus}
            <span className="text-xs ml-1">
              ({Math.round(healthCheckInterval / 1000)}s)
            </span>
          </Badge>

          <Badge variant="outline" className="border-green-500 text-green-500">
            {uploadedFiles.filter((f) => f.status === "analyzed").length}{" "}
            Analyzed
          </Badge>
          <Badge
            variant="outline"
            className="border-orange-500 text-orange-500"
          >
            {processingCount} Processing
          </Badge>
          <Badge
            variant="outline"
            className="border-orange-500 text-orange-500"
          >
            {queuedCount} Queued
          </Badge>
          {errorCount > 0 && (
            <Badge variant="outline" className="border-red-500 text-red-500">
              {errorCount} Errors
            </Badge>
          )}
        </div>
      </div>

      {/* System Status Alerts */}
      {backendStatus !== "healthy" && (
        <Card className="border-red-500/30 bg-red-950/20">
          <CardContent className="py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <WifiOff className="w-5 h-5 text-red-400" />
                <div>
                  <h3 className="text-red-400 font-semibold">
                    Backend Connection Issue
                  </h3>
                  <p className="text-gray-300 text-sm">
                    Analysis services are {backendStatus}. Some features may be
                    limited.
                  </p>
                </div>
              </div>
              <div className="flex gap-2">
                <Button
                  onClick={handleReconnectBackend}
                  variant="outline"
                  size="sm"
                  className="border-green-500 text-green-400"
                  disabled={backendStatus === "connecting"}
                >
                  {backendStatus === "connecting" ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <RefreshCw className="w-4 h-4 mr-2" />
                  )}
                  Reconnect
                </Button>
                <Button
                  onClick={handleRefreshStatus}
                  variant="outline"
                  size="sm"
                  className="border-red-500 text-red-400"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Reset Queue
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* IGN Input Field */}
      <Card className="border-orange-500/20 bg-orange-950/20">
        <CardHeader>
          <CardTitle className="flex items-center justify-center gap-2 text-orange-400">
            <Target className="w-5 h-5" />
            Player Configuration
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center space-y-2">
            <Label htmlFor="player-ign" className="text-orange-300 font-medium">
              In-Game Name (IGN)
            </Label>
            <Input
              id="player-ign"
              type="text"
              placeholder="Enter your MLBB in-game name..."
              value={playerIGN}
              onChange={(e) => setPlayerIGN(e.target.value)}
              className="bg-orange-900/20 border-orange-500/30 text-white placeholder-orange-400/60 focus:border-orange-400 focus:ring-orange-400/20 max-w-md w-full text-center"
            />
            <p className="text-orange-400/80 text-xs text-center max-w-md">
              📋 Required for player profile mapping and MVP/medal detection
              accuracy
              <ClientOnly>
                {playerIGN && (
                  <span className="text-green-400 ml-2">
                    ✅ Saved automatically
                  </span>
                )}
              </ClientOnly>
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Upload Zone */}
      <Card className="border-orange-500/20 bg-orange-950/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-orange-400">
            <Camera className="w-5 h-5" />
            Upload Screenshots
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="border-2 border-dashed border-orange-500/30 rounded-lg p-8 text-center">
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={handleFileUpload}
              className="hidden"
              id="screenshot-upload"
            />
            <label htmlFor="screenshot-upload" className="cursor-pointer">
              <div className="w-16 h-16 mx-auto mb-4 bg-orange-500/20 rounded-full flex items-center justify-center">
                <Image className="w-8 h-8 text-orange-400" />
              </div>
              <h3 className="text-lg font-medium text-orange-400 mb-2">
                Drop Screenshots Here or Click to Browse
              </h3>
              <p className="text-gray-400">
                Supports PNG, JPG, JPEG • Post-match screens, KDA displays,
                medal ceremonies
              </p>
              <p className="text-gray-500 text-xs mt-2">
                Unique IDs generated • No phantom entries • Backend synchronized
              </p>
              {uploadedFiles.length > 0 && (
                <div className="mt-3 space-y-1">
                  <p className="text-orange-400 text-sm font-medium">
                    ✅ {uploadedFiles.length} screenshot
                    {uploadedFiles.length > 1 ? "s" : ""} ready for analysis
                  </p>
                  <ClientOnly>
                    {playerIGN && (
                      <p className="text-orange-400 text-xs">
                        🎮 Player:{" "}
                        <span className="font-medium">{playerIGN}</span>
                      </p>
                    )}
                  </ClientOnly>
                </div>
              )}
            </label>
          </div>
        </CardContent>
      </Card>

      {/* Capabilities Overview */}
      <Card className="border-orange-500/20 bg-orange-950/20">
        <CardContent className="pt-6">
          <div className="text-center space-y-4">
            <h3 className="text-lg font-semibold text-orange-400 mb-4">
              Screenshot Analysis Capabilities
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg">
                <Target className="w-5 h-5 text-orange-500" />
                <div className="text-left">
                  <h4 className="text-white font-medium">
                    Performance Metrics
                  </h4>
                  <p className="text-gray-400 text-sm">
                    KDA, Damage, Gold, GPM
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg">
                <Award className="w-5 h-5 text-yellow-500" />
                <div className="text-left">
                  <h4 className="text-white font-medium">Medal Detection</h4>
                  <p className="text-gray-400 text-sm">
                    MVP, Bronze, Silver, Gold
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg">
                <Brain className="w-5 h-5 text-orange-500" />
                <div className="text-left">
                  <h4 className="text-white font-medium">Hero Analysis</h4>
                  <p className="text-gray-400 text-sm">
                    Role detection, Build analysis
                  </p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Add streamlined Analyze All button after upload */}
      {uploadedFiles.length > 0 &&
        queuedCount > 0 &&
        !isAnalyzing &&
        !analysisComplete && (
          <div className="space-y-4">
            <ClientOnly>
              {!playerIGN.trim() && (
                <div className="text-center p-4 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
                  <p className="text-yellow-400 font-medium">
                    ⚠️ Please enter your In-Game Name (IGN) above to proceed
                    with analysis
                  </p>
                  <p className="text-yellow-300/80 text-sm mt-1">
                    Required for accurate player profile mapping and MVP
                    detection
                  </p>
                </div>
              )}
            </ClientOnly>
            <div className="flex justify-center gap-4">
              <Button
                onClick={handleAnalyzeAll}
                disabled={backendStatus !== "healthy" || !playerIGN.trim()}
                className="bg-orange-600 hover:bg-orange-700 text-lg px-8 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
                size="lg"
              >
                <Brain className="w-5 h-5 mr-2" />
                Analyze All Screenshots ({queuedCount})
              </Button>
              <Button
                onClick={() => {
                  setUploadedFiles([]);
                  setAnalysisResults({});
                  setIsAnalyzing(false);
                  setAnalysisComplete(false);
                  setCompletedResults(null);
                  setIsViewingResults(false);
                }}
                variant="outline"
                className="border-red-500 text-red-400 hover:bg-red-500/10 px-6 py-3"
                size="lg"
              >
                <Trash2 className="w-5 h-5 mr-2" />
                Clear All
              </Button>
            </div>
          </div>
        )}

      {/* NEW: Streamlined Bottom-Center Analysis Status */}
      {(isAnalyzing || analysisComplete) && (
        <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 z-50">
          <div
            className={`
            bg-gradient-to-r backdrop-blur-lg border rounded-2xl px-8 py-6 
            shadow-2xl transition-all duration-700 ease-in-out
            ${
              isAnalyzing
                ? "from-orange-900/80 to-yellow-900/80 border-orange-500/50 animate-pulse"
                : `from-green-900/80 to-blue-900/80 border-green-500/50 scale-105 hover:scale-110 animate-in slide-in-from-bottom-4 fade-in-0 duration-500 ${
                    isViewingResults ? "opacity-50" : "opacity-100"
                  }`
            }
          `}
          >
            {isAnalyzing ? (
              <div className="flex items-center gap-6">
                <div className="w-12 h-12 bg-orange-500/30 rounded-full flex items-center justify-center">
                  <Brain className="w-6 h-6 text-orange-400 animate-pulse" />
                </div>
                <div>
                  <h3 className="text-orange-400 font-bold text-xl">
                    Agent is analyzing...
                  </h3>
                  <p className="text-gray-300 text-sm">
                    Advanced AI analysis in progress
                  </p>
                </div>
                <div className="flex gap-1 ml-4">
                  <div
                    className="w-3 h-3 bg-orange-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0ms" }}
                  ></div>
                  <div
                    className="w-3 h-3 bg-orange-400 rounded-full animate-bounce"
                    style={{ animationDelay: "150ms" }}
                  ></div>
                  <div
                    className="w-3 h-3 bg-orange-400 rounded-full animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  ></div>
                </div>
              </div>
            ) : (
              <button
                onClick={handleViewCompletedResults}
                className="group flex items-center gap-6 hover:scale-105 transition-all duration-500 relative"
              >
                {/* Glowing background effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-green-500/20 to-blue-500/20 rounded-2xl blur-xl opacity-50 group-hover:opacity-80 transition-opacity duration-500"></div>

                {/* Pulse animation ring */}
                <div className="absolute left-4 w-12 h-12 bg-green-400/30 rounded-full animate-ping"></div>

                <div className="relative w-12 h-12 bg-green-500/30 rounded-full flex items-center justify-center group-hover:bg-green-400/40 transition-colors duration-300">
                  <CheckCircle className="w-6 h-6 text-green-400 group-hover:text-green-300" />
                </div>
                <div className="relative">
                  <h3 className="text-green-400 font-bold text-xl group-hover:text-green-300 transition-colors duration-300">
                    View Results
                  </h3>
                  <div className="flex items-center gap-2 text-gray-300 text-sm mt-1">
                    <span>Analysis complete</span>
                    {completedResults?.processingTime && (
                      <>
                        <span>•</span>
                        <span>
                          {completedResults.processingTime.toFixed(1)}s
                        </span>
                      </>
                    )}
                    {completedResults?.confidence && (
                      <>
                        <span>•</span>
                        <span className="text-green-400 font-medium">
                          {completedResults.confidence}% confidence
                        </span>
                      </>
                    )}
                  </div>
                </div>
                <Eye className="w-6 h-6 text-green-400 ml-4 group-hover:text-green-300 group-hover:scale-110 transition-all duration-300" />
              </button>
            )}
          </div>
        </div>
      )}

      {/* Enhanced Results Modal */}
      <ResultsModal />
    </div>
  );
}
