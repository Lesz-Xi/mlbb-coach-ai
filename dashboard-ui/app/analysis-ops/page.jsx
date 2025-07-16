"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Target,
  MapPin,
  Clock,
  Users,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Camera,
  Video,
  Brain,
  Eye,
} from "lucide-react";

export default function AnalysisOpsPage() {
  const [selectedOperation, setSelectedOperation] = useState(null);

  const operations = [
    {
      id: "AO-MVP-001",
      name: "SCREENSHOT MVP DETECTION",
      status: "active",
      priority: "critical",
      type: "Screenshot Analysis",
      player: "SHADOW_STRIKER",
      models: ["YOLO", "EasyOCR", "MVP Detector"],
      progress: 85,
      startTime: "2025-01-17 14:25",
      estimatedCompletion: "2025-01-17 14:32",
      description: "Detecting MVP status from post-match screenshot",
      objectives: [
        "Extract match data",
        "Identify MVP indicator",
        "Generate feedback",
      ],
    },
    {
      id: "AO-SYNC-002",
      name: "TEAM COMP ANALYSIS",
      status: "queued",
      priority: "high",
      type: "Video Analysis",
      player: "FROST_MAGE",
      models: ["Behavioral Model", "Team Sync", "LLM"],
      progress: 15,
      startTime: "2025-01-17 14:30",
      estimatedCompletion: "2025-01-17 14:45",
      description: "Analyzing team composition and synergy patterns",
      objectives: [
        "Extract team data",
        "Analyze synergy",
        "Generate recommendations",
      ],
    },
    {
      id: "AO-MEDAL-003",
      name: "MEDAL RECOGNITION",
      status: "completed",
      priority: "medium",
      type: "Screenshot Analysis",
      player: "BLADE_MASTER",
      models: ["YOLO", "Medal Detector"],
      progress: 100,
      startTime: "2025-01-17 14:15",
      estimatedCompletion: "2025-01-17 14:22",
      description: "Recognizing achievement medals from match results",
      objectives: [
        "Detect medals",
        "Classify achievement",
        "Update player profile",
      ],
    },
    {
      id: "AO-BEHAV-004",
      name: "BEHAVIOR PATTERN ANALYSIS",
      status: "active",
      priority: "high",
      type: "Video Analysis",
      player: "MYSTIC_ARCHER",
      models: ["Behavioral Model", "Pattern Recognition", "LLM"],
      progress: 62,
      startTime: "2025-01-17 14:20",
      estimatedCompletion: "2025-01-17 14:38",
      description: "Analyzing player behavior patterns during teamfights",
      objectives: [
        "Extract behavior data",
        "Identify patterns",
        "Generate insights",
      ],
    },
    {
      id: "AO-PERF-005",
      name: "PERFORMANCE REVIEW",
      status: "failed",
      priority: "medium",
      type: "Screenshot Analysis",
      player: "STORM_RIDER",
      models: ["EasyOCR", "Performance Analyzer"],
      progress: 45,
      startTime: "2025-01-17 14:18",
      estimatedCompletion: "2025-01-17 14:25",
      description: "Comprehensive performance analysis from match statistics",
      objectives: ["OCR match data", "Calculate metrics", "Generate report"],
    },
  ];

  const getStatusColor = (status) => {
    switch (status) {
      case "active":
        return "bg-white/20 text-white";
      case "queued":
        return "bg-orange-500/20 text-orange-500";
      case "completed":
        return "bg-white/20 text-white";
      case "failed":
        return "bg-red-500/20 text-red-500";
      default:
        return "bg-neutral-500/20 text-neutral-300";
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case "critical":
        return "bg-red-500/20 text-red-500";
      case "high":
        return "bg-orange-500/20 text-orange-500";
      case "medium":
        return "bg-neutral-500/20 text-neutral-300";
      case "low":
        return "bg-white/20 text-white";
      default:
        return "bg-neutral-500/20 text-neutral-300";
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "active":
        return <Target className="w-4 h-4" />;
      case "queued":
        return <Clock className="w-4 h-4" />;
      case "completed":
        return <CheckCircle className="w-4 h-4" />;
      case "failed":
        return <XCircle className="w-4 h-4" />;
      default:
        return <AlertTriangle className="w-4 h-4" />;
    }
  };

  const getTypeIcon = (type) => {
    switch (type) {
      case "Screenshot Analysis":
        return <Camera className="w-4 h-4" />;
      case "Video Analysis":
        return <Video className="w-4 h-4" />;
      default:
        return <Eye className="w-4 h-4" />;
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-wider">
            ANALYSIS OPS
          </h1>
          <p className="text-sm text-neutral-400">
            Monitor screenshot and video analysis operations
          </p>
        </div>
        <div className="flex gap-2">
          <Button className="bg-orange-500 hover:bg-orange-600 text-white">
            New Analysis
          </Button>
          <Button className="bg-orange-500 hover:bg-orange-600 text-white">
            Queue Manager
          </Button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-neutral-900 border-neutral-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-neutral-400 tracking-wider">
                  ACTIVE OPS
                </p>
                <p className="text-2xl font-bold text-white font-mono">89</p>
              </div>
              <Target className="w-8 h-8 text-white" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-neutral-900 border-neutral-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-neutral-400 tracking-wider">
                  COMPLETED
                </p>
                <p className="text-2xl font-bold text-white font-mono">2,847</p>
              </div>
              <CheckCircle className="w-8 h-8 text-white" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-neutral-900 border-neutral-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-neutral-400 tracking-wider">
                  FAILED
                </p>
                <p className="text-2xl font-bold text-red-500 font-mono">12</p>
              </div>
              <XCircle className="w-8 h-8 text-red-500" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-neutral-900 border-neutral-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-neutral-400 tracking-wider">
                  SUCCESS RATE
                </p>
                <p className="text-2xl font-bold text-white font-mono">96%</p>
              </div>
              <AlertTriangle className="w-8 h-8 text-white" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Operations List */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {operations.map((operation) => (
          <Card
            key={operation.id}
            className="bg-neutral-900 border-neutral-700 hover:border-orange-500/50 transition-colors cursor-pointer"
            onClick={() => setSelectedOperation(operation)}
          >
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  {getTypeIcon(operation.type)}
                  <div>
                    <CardTitle className="text-sm font-bold text-white tracking-wider">
                      {operation.name}
                    </CardTitle>
                    <p className="text-xs text-neutral-400 font-mono">
                      {operation.id}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {getStatusIcon(operation.status)}
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Badge className={getStatusColor(operation.status)}>
                  {operation.status.toUpperCase()}
                </Badge>
                <Badge className={getPriorityColor(operation.priority)}>
                  {operation.priority.toUpperCase()}
                </Badge>
              </div>

              <p className="text-sm text-neutral-300">
                {operation.description}
              </p>

              <div className="space-y-2">
                <div className="flex items-center gap-2 text-xs text-neutral-400">
                  <Users className="w-3 h-3" />
                  <span>Player: {operation.player}</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-neutral-400">
                  <Brain className="w-3 h-3" />
                  <span>Models: {operation.models.join(", ")}</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-neutral-400">
                  <Clock className="w-3 h-3" />
                  <span>ETA: {operation.estimatedCompletion}</span>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-neutral-400">Progress</span>
                  <span className="text-white font-mono">
                    {operation.progress}%
                  </span>
                </div>
                <div className="w-full bg-neutral-800 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-300 ${
                      operation.status === "failed"
                        ? "bg-red-500"
                        : "bg-orange-500"
                    }`}
                    style={{ width: `${operation.progress}%` }}
                  ></div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Operation Detail Modal */}
      {selectedOperation && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <Card className="bg-neutral-900 border-neutral-700 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <CardHeader className="flex flex-row items-center justify-between">
              <div className="flex items-center gap-3">
                {getTypeIcon(selectedOperation.type)}
                <div>
                  <CardTitle className="text-xl font-bold text-white tracking-wider">
                    {selectedOperation.name}
                  </CardTitle>
                  <p className="text-sm text-neutral-400 font-mono">
                    {selectedOperation.id}
                  </p>
                </div>
              </div>
              <Button
                variant="ghost"
                onClick={() => setSelectedOperation(null)}
                className="text-neutral-400 hover:text-white"
              >
                ✕
              </Button>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <h3 className="text-sm font-medium text-neutral-300 tracking-wider mb-2">
                      OPERATION STATUS
                    </h3>
                    <div className="flex gap-2">
                      <Badge
                        className={getStatusColor(selectedOperation.status)}
                      >
                        {selectedOperation.status.toUpperCase()}
                      </Badge>
                      <Badge
                        className={getPriorityColor(selectedOperation.priority)}
                      >
                        {selectedOperation.priority.toUpperCase()}
                      </Badge>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-sm font-medium text-neutral-300 tracking-wider mb-2">
                      OPERATION DETAILS
                    </h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-neutral-400">Type:</span>
                        <span className="text-white">
                          {selectedOperation.type}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-neutral-400">Player:</span>
                        <span className="text-white font-mono">
                          {selectedOperation.player}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-neutral-400">Started:</span>
                        <span className="text-white font-mono">
                          {selectedOperation.startTime}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-neutral-400">ETA:</span>
                        <span className="text-white font-mono">
                          {selectedOperation.estimatedCompletion}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <h3 className="text-sm font-medium text-neutral-300 tracking-wider mb-2">
                      PROGRESS
                    </h3>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-neutral-400">Completion</span>
                        <span className="text-white font-mono">
                          {selectedOperation.progress}%
                        </span>
                      </div>
                      <div className="w-full bg-neutral-800 rounded-full h-3">
                        <div
                          className={`h-3 rounded-full transition-all duration-300 ${
                            selectedOperation.status === "failed"
                              ? "bg-red-500"
                              : "bg-orange-500"
                          }`}
                          style={{ width: `${selectedOperation.progress}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-sm font-medium text-neutral-300 tracking-wider mb-2">
                      MODELS INVOLVED
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {selectedOperation.models.map((model) => (
                        <Badge
                          key={model}
                          className="bg-neutral-800 text-neutral-300"
                        >
                          {model}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-sm font-medium text-neutral-300 tracking-wider mb-2">
                  OBJECTIVES
                </h3>
                <div className="space-y-2">
                  {selectedOperation.objectives.map((objective, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-2 text-sm"
                    >
                      <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                      <span className="text-neutral-300">{objective}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-sm font-medium text-neutral-300 tracking-wider mb-2">
                  DESCRIPTION
                </h3>
                <p className="text-sm text-neutral-300">
                  {selectedOperation.description}
                </p>
              </div>

              <div className="flex gap-2 pt-4 border-t border-neutral-700">
                <Button className="bg-orange-500 hover:bg-orange-600 text-white">
                  {selectedOperation.status === "failed"
                    ? "Retry Operation"
                    : "View Details"}
                </Button>
                <Button
                  variant="outline"
                  className="border-neutral-700 text-neutral-400 hover:bg-neutral-800 hover:text-neutral-300 bg-transparent"
                >
                  View Logs
                </Button>
                <Button
                  variant="outline"
                  className="border-neutral-700 text-neutral-400 hover:bg-neutral-800 hover:text-neutral-300 bg-transparent"
                >
                  Cancel Operation
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
