import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const ValidationDashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("overview");
  const [validationEntries, setValidationEntries] = useState([]);
  const [selectedEntry, setSelectedEntry] = useState(null);
  const [edgeCaseResults, setEdgeCaseResults] = useState([]);

  // Colors for charts
  const COLORS = ["#8884d8", "#82ca9d", "#ffc658", "#ff7c7c", "#8dd1e1"];

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const response = await fetch("/api/validation-dashboard/");
      const data = await response.json();
      if (data.success) {
        setDashboardData(data.dashboard_data);
      }
    } catch (error) {
      console.error("Failed to fetch dashboard data:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchValidationEntries = async () => {
    try {
      const response = await fetch(
        "/api/validation-entries/?status=pending&limit=50"
      );
      const data = await response.json();
      if (data.success) {
        setValidationEntries(data.entries);
      }
    } catch (error) {
      console.error("Failed to fetch validation entries:", error);
    }
  };

  const submitFeedback = async (entryId, feedbackData) => {
    try {
      const response = await fetch("/api/report-feedback/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          entry_id: entryId,
          ...feedbackData,
        }),
      });

      if (response.ok) {
        // Refresh data
        fetchDashboardData();
        fetchValidationEntries();
      }
    } catch (error) {
      console.error("Failed to submit feedback:", error);
    }
  };

  const submitAnnotation = async (entryId, annotationData) => {
    try {
      const response = await fetch("/api/annotate/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          entry_id: entryId,
          ...annotationData,
        }),
      });

      if (response.ok) {
        fetchDashboardData();
        fetchValidationEntries();
        setSelectedEntry(null);
      }
    } catch (error) {
      console.error("Failed to submit annotation:", error);
    }
  };

  const runEdgeCaseTest = async (testData) => {
    try {
      const formData = new FormData();
      formData.append("test_name", testData.name);
      formData.append("test_description", testData.description);
      formData.append("test_category", testData.category);

      testData.files.forEach((file) => {
        formData.append("test_files", file);
      });

      const response = await fetch("/api/edge-case-test/", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      if (result.success) {
        setEdgeCaseResults((prev) => [result, ...prev]);
      }
    } catch (error) {
      console.error("Failed to run edge case test:", error);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-mlbb-gold"></div>
        <span className="ml-2 text-gray-300">
          Loading validation dashboard...
        </span>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 text-white min-h-screen p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-mlbb-gold mb-2">
          🔬 Real-User Validation Dashboard
        </h1>
        <p className="text-gray-300">
          Monitor system performance, collect feedback, and test edge cases
        </p>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 mb-6 bg-gray-800 p-1 rounded-lg">
        {[
          { id: "overview", label: "📊 Overview", icon: "📊" },
          { id: "metrics", label: "📈 Metrics", icon: "📈" },
          { id: "annotation", label: "✏️ Annotation", icon: "✏️" },
          { id: "edgecases", label: "🧪 Edge Cases", icon: "🧪" },
          { id: "feedback", label: "💬 Feedback", icon: "💬" },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-md transition-colors ${
              activeTab === tab.id
                ? "bg-mlbb-gold text-mlbb-dark font-semibold"
                : "text-gray-300 hover:text-white hover:bg-gray-700"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === "overview" && dashboardData && (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gray-800 p-6 rounded-lg">
              <h3 className="text-sm font-medium text-gray-400">
                Total Validations
              </h3>
              <p className="text-3xl font-bold text-mlbb-gold">
                {dashboardData.total_validations}
              </p>
            </div>

            <div className="bg-gray-800 p-6 rounded-lg">
              <h3 className="text-sm font-medium text-gray-400">
                Accuracy Rate
              </h3>
              <p className="text-3xl font-bold text-green-400">
                {(dashboardData.accuracy_rate * 100).toFixed(1)}%
              </p>
            </div>

            <div className="bg-gray-800 p-6 rounded-lg">
              <h3 className="text-sm font-medium text-gray-400">
                Avg Confidence
              </h3>
              <p className="text-3xl font-bold text-blue-400">
                {(dashboardData.avg_confidence * 100).toFixed(1)}%
              </p>
            </div>

            <div className="bg-gray-800 p-6 rounded-lg">
              <h3 className="text-sm font-medium text-gray-400">
                User Satisfaction
              </h3>
              <p className="text-3xl font-bold text-purple-400">
                {dashboardData.user_satisfaction.toFixed(1)}/5
              </p>
            </div>
          </div>

          {/* System Alerts */}
          {dashboardData.system_alerts &&
            dashboardData.system_alerts.length > 0 && (
              <div className="bg-red-900 border border-red-700 p-4 rounded-lg">
                <h3 className="text-lg font-semibold text-red-400 mb-2">
                  🚨 System Alerts
                </h3>
                <ul className="space-y-1">
                  {dashboardData.system_alerts.map((alert, index) => (
                    <li key={index} className="text-red-300">
                      • {alert}
                    </li>
                  ))}
                </ul>
              </div>
            )}

          {/* Performance Warnings */}
          {dashboardData.performance_warnings &&
            dashboardData.performance_warnings.length > 0 && (
              <div className="bg-yellow-900 border border-yellow-700 p-4 rounded-lg">
                <h3 className="text-lg font-semibold text-yellow-400 mb-2">
                  ⚠️ Performance Warnings
                </h3>
                <ul className="space-y-1">
                  {dashboardData.performance_warnings.map((warning, index) => (
                    <li key={index} className="text-yellow-300">
                      • {warning}
                    </li>
                  ))}
                </ul>
              </div>
            )}

          {/* Accuracy Over Time Chart */}
          {dashboardData.accuracy_over_time && (
            <div className="bg-gray-800 p-6 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">📈 Accuracy Trend</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={dashboardData.accuracy_over_time}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="date" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1F2937",
                      border: "1px solid #374151",
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#10B981"
                    strokeWidth={2}
                    name="Accuracy %"
                  />
                  <Line
                    type="monotone"
                    dataKey="confidence"
                    stroke="#3B82F6"
                    strokeWidth={2}
                    name="Confidence %"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Metrics Tab */}
      {activeTab === "metrics" && dashboardData && (
        <div className="space-y-6">
          {/* Device Performance */}
          {dashboardData.device_performance && (
            <div className="bg-gray-800 p-6 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">
                📱 Performance by Device
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={dashboardData.device_performance}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="device" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1F2937",
                      border: "1px solid #374151",
                    }}
                  />
                  <Bar dataKey="accuracy" fill="#10B981" name="Accuracy %" />
                  <Bar
                    dataKey="confidence"
                    fill="#3B82F6"
                    name="Confidence %"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Confidence Distribution */}
          {dashboardData.confidence_distribution && (
            <div className="bg-gray-800 p-6 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">
                🎯 Confidence Distribution
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={dashboardData.confidence_distribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) =>
                      `${name}: ${(percent * 100).toFixed(0)}%`
                    }
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {dashboardData.confidence_distribution.map(
                      (entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={COLORS[index % COLORS.length]}
                        />
                      )
                    )}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Edge Case Frequency */}
          {dashboardData.edge_case_frequency && (
            <div className="bg-gray-800 p-6 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">
                🧪 Edge Case Frequency
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={dashboardData.edge_case_frequency}
                  layout="horizontal"
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis type="number" stroke="#9CA3AF" />
                  <YAxis
                    dataKey="case"
                    type="category"
                    stroke="#9CA3AF"
                    width={100}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1F2937",
                      border: "1px solid #374151",
                    }}
                  />
                  <Bar dataKey="frequency" fill="#F59E0B" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Annotation Tab */}
      {activeTab === "annotation" && (
        <AnnotationInterface
          entries={validationEntries}
          selectedEntry={selectedEntry}
          onSelectEntry={setSelectedEntry}
          onSubmitAnnotation={submitAnnotation}
          onFetchEntries={fetchValidationEntries}
        />
      )}

      {/* Edge Cases Tab */}
      {activeTab === "edgecases" && (
        <EdgeCaseTestInterface
          results={edgeCaseResults}
          onRunTest={runEdgeCaseTest}
        />
      )}

      {/* Feedback Tab */}
      {activeTab === "feedback" && dashboardData && (
        <FeedbackInterface
          recentFeedback={dashboardData.recent_feedback}
          onSubmitFeedback={submitFeedback}
        />
      )}
    </div>
  );
};

// Annotation Interface Component
const AnnotationInterface = ({
  entries,
  selectedEntry,
  onSelectEntry,
  onSubmitAnnotation,
  onFetchEntries,
}) => {
  const [annotationForm, setAnnotationForm] = useState({});

  useEffect(() => {
    onFetchEntries();
  }, [onFetchEntries]);

  const handleAnnotationSubmit = (e) => {
    e.preventDefault();
    if (selectedEntry) {
      onSubmitAnnotation(selectedEntry.entry_id, annotationForm);
      setAnnotationForm({});
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800 p-6 rounded-lg">
        <h3 className="text-lg font-semibold mb-4">
          ✏️ Manual Annotation Interface
        </h3>

        {/* Entry Selection */}
        <div className="mb-6">
          <h4 className="font-medium mb-2">Pending Validation Entries</h4>
          <div className="max-h-60 overflow-y-auto space-y-2">
            {entries.length === 0 ? (
              <p className="text-gray-400 italic">
                No pending entries for annotation
              </p>
            ) : (
              entries.map((entry) => (
                <div
                  key={entry.entry_id}
                  onClick={() => onSelectEntry(entry)}
                  className={`p-3 rounded border cursor-pointer transition-colors ${
                    selectedEntry?.entry_id === entry.entry_id
                      ? "border-mlbb-gold bg-yellow-900"
                      : "border-gray-600 bg-gray-700 hover:bg-gray-600"
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <span className="font-medium">
                      {entry.screenshot_metadata?.filename}
                    </span>
                    <span className="text-sm text-gray-400">
                      Confidence:{" "}
                      {(
                        entry.ai_confidence_scores?.overall_confidence * 100 ||
                        0
                      ).toFixed(1)}
                      %
                    </span>
                  </div>
                  <div className="text-sm text-gray-300 mt-1">
                    Device: {entry.screenshot_metadata?.device_type} | Locale:{" "}
                    {entry.screenshot_metadata?.game_locale}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Annotation Form */}
        {selectedEntry && (
          <form onSubmit={handleAnnotationSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">
                  Player IGN
                </label>
                <input
                  type="text"
                  value={annotationForm.player_ign || ""}
                  onChange={(e) =>
                    setAnnotationForm({
                      ...annotationForm,
                      player_ign: e.target.value,
                    })
                  }
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Hero Played
                </label>
                <input
                  type="text"
                  value={annotationForm.hero_played || ""}
                  onChange={(e) =>
                    setAnnotationForm({
                      ...annotationForm,
                      hero_played: e.target.value,
                    })
                  }
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Kills</label>
                <input
                  type="number"
                  value={annotationForm.kills || ""}
                  onChange={(e) =>
                    setAnnotationForm({
                      ...annotationForm,
                      kills: parseInt(e.target.value),
                    })
                  }
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Deaths</label>
                <input
                  type="number"
                  value={annotationForm.deaths || ""}
                  onChange={(e) =>
                    setAnnotationForm({
                      ...annotationForm,
                      deaths: parseInt(e.target.value),
                    })
                  }
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Assists
                </label>
                <input
                  type="number"
                  value={annotationForm.assists || ""}
                  onChange={(e) =>
                    setAnnotationForm({
                      ...annotationForm,
                      assists: parseInt(e.target.value),
                    })
                  }
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">
                  Hero Damage
                </label>
                <input
                  type="number"
                  value={annotationForm.hero_damage || ""}
                  onChange={(e) =>
                    setAnnotationForm({
                      ...annotationForm,
                      hero_damage: parseInt(e.target.value),
                    })
                  }
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Teamfight Participation (%)
                </label>
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={annotationForm.teamfight_participation || ""}
                  onChange={(e) =>
                    setAnnotationForm({
                      ...annotationForm,
                      teamfight_participation: parseInt(e.target.value),
                    })
                  }
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">
                  Match Result
                </label>
                <select
                  value={annotationForm.match_result || ""}
                  onChange={(e) =>
                    setAnnotationForm({
                      ...annotationForm,
                      match_result: e.target.value,
                    })
                  }
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                  required
                >
                  <option value="">Select Result</option>
                  <option value="Victory">Victory</option>
                  <option value="Defeat">Defeat</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Match Duration (minutes)
                </label>
                <input
                  type="number"
                  value={annotationForm.match_duration_minutes || ""}
                  onChange={(e) =>
                    setAnnotationForm({
                      ...annotationForm,
                      match_duration_minutes: parseInt(e.target.value),
                    })
                  }
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Annotation Notes
              </label>
              <textarea
                value={annotationForm.annotator_notes || ""}
                onChange={(e) =>
                  setAnnotationForm({
                    ...annotationForm,
                    annotator_notes: e.target.value,
                  })
                }
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                rows="3"
                placeholder="Optional notes about this annotation..."
              />
            </div>

            <button
              type="submit"
              className="bg-mlbb-gold text-mlbb-dark px-6 py-2 rounded font-semibold hover:bg-yellow-400 transition-colors"
            >
              Submit Annotation
            </button>
          </form>
        )}
      </div>
    </div>
  );
};

// Edge Case Test Interface Component
const EdgeCaseTestInterface = ({ results, onRunTest }) => {
  const [testForm, setTestForm] = useState({
    name: "",
    description: "",
    category: "",
    files: [],
  });

  const handleTestSubmit = (e) => {
    e.preventDefault();
    if (testForm.files.length > 0) {
      onRunTest(testForm);
      setTestForm({ name: "", description: "", category: "", files: [] });
    }
  };

  const handleFileChange = (e) => {
    setTestForm({ ...testForm, files: Array.from(e.target.files) });
  };

  return (
    <div className="space-y-6">
      {/* Test Runner */}
      <div className="bg-gray-800 p-6 rounded-lg">
        <h3 className="text-lg font-semibold mb-4">🧪 Edge Case Test Runner</h3>

        <form onSubmit={handleTestSubmit} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">
                Test Name
              </label>
              <input
                type="text"
                value={testForm.name}
                onChange={(e) =>
                  setTestForm({ ...testForm, name: e.target.value })
                }
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                placeholder="e.g., Low Resolution Test"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Category</label>
              <select
                value={testForm.category}
                onChange={(e) =>
                  setTestForm({ ...testForm, category: e.target.value })
                }
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                required
              >
                <option value="">Select Category</option>
                <option value="resolution">Resolution</option>
                <option value="locale">Locale</option>
                <option value="device">Device</option>
                <option value="ui_variation">UI Variation</option>
                <option value="compression">Compression</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              Description
            </label>
            <textarea
              value={testForm.description}
              onChange={(e) =>
                setTestForm({ ...testForm, description: e.target.value })
              }
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
              rows="2"
              placeholder="Describe what this test is designed to validate..."
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              Test Screenshots
            </label>
            <input
              type="file"
              multiple
              accept="image/*"
              onChange={handleFileChange}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
              required
            />
            {testForm.files.length > 0 && (
              <p className="text-sm text-gray-400 mt-1">
                {testForm.files.length} file(s) selected
              </p>
            )}
          </div>

          <button
            type="submit"
            className="bg-blue-600 text-white px-6 py-2 rounded font-semibold hover:bg-blue-700 transition-colors"
          >
            Run Edge Case Test
          </button>
        </form>
      </div>

      {/* Test Results */}
      <div className="bg-gray-800 p-6 rounded-lg">
        <h3 className="text-lg font-semibold mb-4">📊 Test Results</h3>

        {results.length === 0 ? (
          <p className="text-gray-400 italic">
            No test results yet. Run a test to see results here.
          </p>
        ) : (
          <div className="space-y-4">
            {results.map((result, index) => (
              <div key={index} className="border border-gray-600 rounded p-4">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="font-semibold">{result.test_name}</h4>
                  <div className="flex space-x-4 text-sm">
                    <span
                      className={`px-2 py-1 rounded ${
                        result.success_rate > 0.8
                          ? "bg-green-900 text-green-300"
                          : result.success_rate > 0.5
                          ? "bg-yellow-900 text-yellow-300"
                          : "bg-red-900 text-red-300"
                      }`}
                    >
                      Success: {(result.success_rate * 100).toFixed(1)}%
                    </span>
                    <span className="text-gray-400">
                      Avg Confidence: {(result.avg_confidence * 100).toFixed(1)}
                      %
                    </span>
                  </div>
                </div>

                <p className="text-gray-300 text-sm mb-2">
                  Category: {result.test_category}
                </p>
                <p className="text-gray-300 text-sm mb-3">
                  Files: {result.total_files}
                </p>

                {/* Individual file results */}
                <div className="space-y-1">
                  {result.results.map((fileResult, fileIndex) => (
                    <div
                      key={fileIndex}
                      className="flex justify-between items-center text-sm"
                    >
                      <span className="text-gray-400">
                        {fileResult.filename}
                      </span>
                      <div className="flex space-x-2">
                        {fileResult.success ? (
                          <>
                            <span className="text-green-400">✓</span>
                            <span className="text-gray-300">
                              {(fileResult.confidence * 100).toFixed(1)}%
                            </span>
                          </>
                        ) : (
                          <span className="text-red-400">✗ Failed</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// Feedback Interface Component
const FeedbackInterface = ({ recentFeedback }) => {
  return (
    <div className="space-y-6">
      <div className="bg-gray-800 p-6 rounded-lg">
        <h3 className="text-lg font-semibold mb-4">💬 Recent User Feedback</h3>

        {!recentFeedback || recentFeedback.length === 0 ? (
          <p className="text-gray-400 italic">No recent feedback available</p>
        ) : (
          <div className="space-y-4">
            {recentFeedback.map((feedback, index) => (
              <div key={index} className="border border-gray-600 rounded p-4">
                <div className="flex justify-between items-start mb-2">
                  <div className="flex items-center space-x-2">
                    <span
                      className={`w-3 h-3 rounded-full ${
                        feedback.is_analysis_correct
                          ? "bg-green-400"
                          : "bg-red-400"
                      }`}
                    ></span>
                    <span className="font-medium">
                      {feedback.is_analysis_correct
                        ? "Correct Analysis"
                        : "Incorrect Analysis"}
                    </span>
                  </div>
                  <span className="text-sm text-gray-400">
                    Rating: {feedback.ease_of_use_rating}/5
                  </span>
                </div>

                {feedback.user_comments && (
                  <p className="text-gray-300 text-sm mb-2">
                    &quot;{feedback.user_comments}&quot;
                  </p>
                )}

                {feedback.incorrect_fields &&
                  feedback.incorrect_fields.length > 0 && (
                    <div className="text-sm">
                      <span className="text-red-400">Incorrect fields: </span>
                      <span className="text-gray-300">
                        {feedback.incorrect_fields.join(", ")}
                      </span>
                    </div>
                  )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ValidationDashboard;
