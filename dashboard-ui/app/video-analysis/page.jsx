"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw,
  Clock,
  Target,
  Users,
  Brain,
  TrendingUp,
  MapPin,
  Activity,
  Zap,
  Eye,
  Download,
  Settings,
} from "lucide-react";

export default function VideoAnalysisPage() {
  // Video player state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);

  // Analysis state
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [selectedEvent, setSelectedEvent] = useState(null);

  // Video file state
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const videoRef = useRef(null);

  // Mock analysis results for demonstration
  const mockAnalysisResults = {
    playerIGN: "Lesz XVII",
    videoDuration: 847, // seconds
    gamePhases: {
      early: { start: 0, end: 300, performance: 85 },
      mid: { start: 300, end: 600, performance: 72 },
      late: { start: 600, end: 847, performance: 91 },
    },
    events: [
      {
        timestamp: 45,
        type: "first_blood",
        player: "Lesz XVII",
        confidence: 0.95,
      },
      { timestamp: 127, type: "tower_destroy", team: "ally", confidence: 0.89 },
      {
        timestamp: 234,
        type: "lord_attempt",
        result: "success",
        confidence: 0.92,
      },
      {
        timestamp: 456,
        type: "team_fight",
        outcome: "victory",
        confidence: 0.88,
      },
      { timestamp: 678, type: "turtle_secure", team: "ally", confidence: 0.94 },
      { timestamp: 789, type: "maniac", player: "Lesz XVII", confidence: 0.97 },
    ],
    behavioralInsights: {
      playstyle: "aggressive-roamer",
      riskProfile: "calculated",
      gameTempo: "early-game-focused",
      teamCoordination: 0.87,
      positioningScore: 0.82,
      reactionTime: 0.91,
    },
    heatmapData: {
      kills: [
        [100, 150],
        [250, 300],
        [400, 200],
      ],
      deaths: [
        [180, 280],
        [350, 150],
      ],
      assists: [
        [120, 200],
        [280, 250],
        [380, 320],
        [450, 180],
      ],
    },
  };

  const handleVideoUpload = useCallback((event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith("video/")) {
      setVideoFile(file);
      const url = URL.createObjectURL(file);
      setVideoUrl(url);
    }
  }, []);

  const handleAnalyzeVideo = useCallback(async () => {
    if (!videoFile) return;

    setIsAnalyzing(true);
    setAnalysisProgress(0);

    // Simulate analysis progress
    const interval = setInterval(() => {
      setAnalysisProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsAnalyzing(false);
          setAnalysisResults(mockAnalysisResults);
          return 100;
        }
        return prev + 2;
      });
    }, 100);
  }, [videoFile]);

  const handleTimelineSeek = useCallback((timestamp) => {
    if (videoRef.current) {
      videoRef.current.currentTime = timestamp;
      setCurrentTime(timestamp);
    }
  }, []);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const getPhaseColor = (phase) => {
    switch (phase) {
      case "early":
        return "bg-green-500";
      case "mid":
        return "bg-yellow-500";
      case "late":
        return "bg-purple-500";
      default:
        return "bg-gray-500";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-orange-900 to-yellow-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-orange-400 to-yellow-400 bg-clip-text text-transparent">
              Video Analysis Laboratory
            </h1>
            <p className="text-gray-300 mt-2">
              Temporal Intelligence • Behavioral Flow Mapping • Strategic
              Analysis
            </p>
          </div>
          <div className="flex gap-3">
            <Button
              variant="outline"
              className="border-orange-500 text-orange-400"
            >
              <Download className="w-4 h-4 mr-2" />
              Export Analysis
            </Button>
            <Button
              variant="outline"
              className="border-yellow-500 text-yellow-400"
            >
              <Settings className="w-4 h-4 mr-2" />
              Pipeline Settings
            </Button>
          </div>
        </div>

        {/* Video Upload Section */}
        {!videoFile && (
          <Card className="border-orange-500/20 bg-orange-950/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-orange-400">
                <Play className="w-5 h-5" />
                Upload Match Replay
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="border-2 border-dashed border-orange-500/30 rounded-lg p-8 text-center">
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleVideoUpload}
                  className="hidden"
                  id="video-upload"
                />
                <label htmlFor="video-upload" className="cursor-pointer">
                  <div className="w-16 h-16 mx-auto mb-4 bg-orange-500/20 rounded-full flex items-center justify-center">
                    <Play className="w-8 h-8 text-orange-400" />
                  </div>
                  <h3 className="text-lg font-medium text-orange-400 mb-2">
                    Upload MLBB Match Video
                  </h3>
                  <p className="text-gray-400">
                    Supports MP4, AVI, MOV formats • Max 2GB
                  </p>
                </label>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Video Player & Analysis Interface */}
        {videoFile && (
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            {/* Main Video Player */}
            <div className="xl:col-span-2">
              <Card className="border-orange-500/20 bg-orange-950/20">
                <CardContent className="p-0">
                  <div className="relative">
                    <video
                      ref={videoRef}
                      src={videoUrl}
                      className="w-full h-64 sm:h-80 object-cover rounded-t-lg"
                      onTimeUpdate={(e) => setCurrentTime(e.target.currentTime)}
                      onLoadedMetadata={(e) => setDuration(e.target.duration)}
                    />

                    {/* Video Controls Overlay */}
                    <div className="absolute bottom-0 left-0 right-0 bg-black/80 p-4">
                      <div className="flex items-center gap-4">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => {
                            if (isPlaying) {
                              videoRef.current?.pause();
                            } else {
                              videoRef.current?.play();
                            }
                            setIsPlaying(!isPlaying);
                          }}
                        >
                          {isPlaying ? (
                            <Pause className="w-4 h-4" />
                          ) : (
                            <Play className="w-4 h-4" />
                          )}
                        </Button>

                        <div className="flex-1">
                          <Slider
                            value={[currentTime]}
                            max={duration}
                            step={1}
                            onValueChange={([value]) =>
                              handleTimelineSeek(value)
                            }
                            className="w-full"
                          />
                        </div>

                        <span className="text-sm text-gray-300 min-w-20">
                          {formatTime(currentTime)} / {formatTime(duration)}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Timeline with Events */}
                  {analysisResults && (
                    <div className="p-4 border-t border-orange-500/20">
                      <h3 className="text-sm font-medium text-orange-400 mb-3">
                        Match Timeline
                      </h3>
                      <div className="relative h-8 bg-gray-800 rounded-lg overflow-hidden">
                        {/* Game Phase Backgrounds */}
                        {Object.entries(analysisResults.gamePhases).map(
                          ([phase, data]) => (
                            <div
                              key={phase}
                              className={`absolute top-0 h-full ${getPhaseColor(
                                phase
                              )} opacity-30`}
                              style={{
                                left: `${(data.start / duration) * 100}%`,
                                width: `${
                                  ((data.end - data.start) / duration) * 100
                                }%`,
                              }}
                            />
                          )
                        )}

                        {/* Event Markers */}
                        {analysisResults.events.map((event, index) => (
                          <div
                            key={index}
                            className="absolute top-1 w-1.5 h-6 bg-yellow-400 rounded-full cursor-pointer hover:bg-yellow-300 transition-colors"
                            style={{
                              left: `${(event.timestamp / duration) * 100}%`,
                            }}
                            onClick={() => handleTimelineSeek(event.timestamp)}
                            title={`${event.type} at ${formatTime(
                              event.timestamp
                            )}`}
                          />
                        ))}

                        {/* Current Time Indicator */}
                        <div
                          className="absolute top-0 w-0.5 h-full bg-white"
                          style={{ left: `${(currentTime / duration) * 100}%` }}
                        />
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Analysis Button */}
              {!analysisResults && !isAnalyzing && (
                <Card className="border-orange-500/20 bg-orange-950/20 mt-4">
                  <CardContent className="p-4">
                    <Button
                      onClick={handleAnalyzeVideo}
                      className="w-full bg-gradient-to-r from-orange-600 to-yellow-600 hover:from-orange-700 hover:to-yellow-700"
                    >
                      <Brain className="w-4 h-4 mr-2" />
                      Analyze Video with AI
                    </Button>
                  </CardContent>
                </Card>
              )}

              {/* Analysis Progress */}
              {isAnalyzing && (
                <Card className="border-yellow-500/20 bg-yellow-950/20 mt-4">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3 mb-3">
                      <Activity className="w-5 h-5 text-yellow-400 animate-pulse" />
                      <span className="text-yellow-400 font-medium">
                        Analyzing Video... {analysisProgress}%
                      </span>
                    </div>
                    <Progress value={analysisProgress} className="w-full" />
                    <p className="text-sm text-gray-400 mt-2">
                      Processing temporal events, behavioral patterns, and
                      strategic insights...
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>

            {/* Analysis Sidebar */}
            <div className="space-y-6">
              {/* Quick Stats */}
              {analysisResults && (
                <Card className="border-orange-500/20 bg-orange-950/20">
                  <CardHeader>
                    <CardTitle className="text-orange-400 text-lg">
                      Analysis Overview
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">Player:</span>
                      <Badge
                        variant="outline"
                        className="border-orange-500 text-orange-400"
                      >
                        {analysisResults.playerIGN}
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">Duration:</span>
                      <span className="text-white">
                        {formatTime(analysisResults.videoDuration)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">Events:</span>
                      <span className="text-white">
                        {analysisResults.events.length}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Game Phases */}
              {analysisResults && (
                <Card className="border-yellow-500/20 bg-yellow-950/20">
                  <CardHeader>
                    <CardTitle className="text-yellow-400 text-lg flex items-center gap-2">
                      <Clock className="w-5 h-5" />
                      Game Phases
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {Object.entries(analysisResults.gamePhases).map(
                      ([phase, data]) => (
                        <div key={phase} className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-gray-300 capitalize">
                              {phase} Game:
                            </span>
                            <span className="text-white">
                              {data.performance}%
                            </span>
                          </div>
                          <Progress
                            value={data.performance}
                            className="w-full h-2"
                          />
                        </div>
                      )
                    )}
                  </CardContent>
                </Card>
              )}

              {/* Behavioral Insights */}
              {analysisResults && (
                <Card className="border-orange-500/20 bg-orange-950/20">
                  <CardHeader>
                    <CardTitle className="text-orange-400 text-lg flex items-center gap-2">
                      <Brain className="w-5 h-5" />
                      Behavioral Profile
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">Playstyle:</span>
                      <Badge className="bg-orange-600 text-white">
                        {analysisResults.behavioralInsights.playstyle}
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">Risk Profile:</span>
                      <span className="text-white">
                        {analysisResults.behavioralInsights.riskProfile}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">Team Coordination:</span>
                      <span className="text-white">
                        {Math.round(
                          analysisResults.behavioralInsights.teamCoordination *
                            100
                        )}
                        %
                      </span>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        )}

        {/* Detailed Analysis Tabs */}
        {analysisResults && (
          <Card className="border-gray-500/20 bg-gray-950/20">
            <CardContent className="p-0">
              <Tabs defaultValue="events" className="w-full">
                <TabsList className="grid w-full grid-cols-4 bg-gray-800/50">
                  <TabsTrigger
                    value="events"
                    className="flex items-center gap-2"
                  >
                    <Target className="w-4 h-4" />
                    Events
                  </TabsTrigger>
                  <TabsTrigger
                    value="heatmap"
                    className="flex items-center gap-2"
                  >
                    <MapPin className="w-4 h-4" />
                    Heatmap
                  </TabsTrigger>
                  <TabsTrigger
                    value="behavior"
                    className="flex items-center gap-2"
                  >
                    <TrendingUp className="w-4 h-4" />
                    Behavior
                  </TabsTrigger>
                  <TabsTrigger
                    value="coaching"
                    className="flex items-center gap-2"
                  >
                    <Brain className="w-4 h-4" />
                    Coaching
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="events" className="p-6 space-y-4">
                  <h3 className="text-xl font-semibold text-white mb-4">
                    Key Events Timeline
                  </h3>
                  {analysisResults.events.map((event, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg cursor-pointer hover:bg-gray-700/50 transition-colors"
                      onClick={() => handleTimelineSeek(event.timestamp)}
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                        <span className="text-white font-medium">
                          {event.type.replace("_", " ")}
                        </span>
                        <span className="text-gray-400">
                          {formatTime(event.timestamp)}
                        </span>
                      </div>
                      <Badge
                        variant="outline"
                        className="border-orange-500 text-orange-400"
                      >
                        {Math.round(event.confidence * 100)}% confidence
                      </Badge>
                    </div>
                  ))}
                </TabsContent>

                <TabsContent value="heatmap" className="p-6">
                  <h3 className="text-xl font-semibold text-white mb-4">
                    Position Heatmap
                  </h3>
                  <div className="bg-gray-800 rounded-lg p-6 h-64 flex items-center justify-center">
                    <p className="text-gray-400">
                      Heatmap visualization will be rendered here
                    </p>
                  </div>
                </TabsContent>

                <TabsContent value="behavior" className="p-6">
                  <h3 className="text-xl font-semibold text-white mb-4">
                    Behavioral Analysis
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card className="border-orange-500/20 bg-orange-950/20">
                      <CardHeader>
                        <CardTitle className="text-orange-400">
                          Decision Making
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <Progress value={85} className="w-full mb-2" />
                        <p className="text-sm text-gray-400">
                          Strong tactical decisions in team fights
                        </p>
                      </CardContent>
                    </Card>
                    <Card className="border-yellow-500/20 bg-yellow-950/20">
                      <CardHeader>
                        <CardTitle className="text-yellow-400">
                          Positioning
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <Progress value={82} className="w-full mb-2" />
                        <p className="text-sm text-gray-400">
                          Excellent map awareness and spacing
                        </p>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>

                <TabsContent value="coaching" className="p-6">
                  <h3 className="text-xl font-semibold text-white mb-4">
                    AI Coaching Insights
                  </h3>
                  <div className="space-y-4">
                    <Card className="border-orange-500/20 bg-orange-950/20">
                      <CardContent className="p-4">
                        <h4 className="text-orange-400 font-medium mb-2">
                          ✓ Strengths
                        </h4>
                        <p className="text-gray-300">
                          Excellent early game aggression and objective control
                        </p>
                      </CardContent>
                    </Card>
                    <Card className="border-yellow-500/20 bg-yellow-950/20">
                      <CardContent className="p-4">
                        <h4 className="text-yellow-400 font-medium mb-2">
                          ⚠ Areas for Improvement
                        </h4>
                        <p className="text-gray-300">
                          Late game positioning could be more conservative
                        </p>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
