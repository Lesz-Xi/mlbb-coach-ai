"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function PlayerHubPage() {
  return (
    <div className="p-6 space-y-6">
      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Player Distribution Overview */}
        <Card className="lg:col-span-4 bg-neutral-900 border-neutral-700">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-neutral-300 tracking-wider">
              PLAYER DISTRIBUTION
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-white font-mono">
                  356
                </div>
                <div className="text-xs text-neutral-500">Active Players</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-white font-mono">
                  89
                </div>
                <div className="text-xs text-neutral-500">In Review</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-white font-mono">
                  124
                </div>
                <div className="text-xs text-neutral-500">Training Mode</div>
              </div>
            </div>

            <div className="space-y-2">
              {[
                {
                  id: "P-001",
                  name: "SHADOW_STRIKER",
                  status: "analyzed",
                  game: "MLBB",
                },
                {
                  id: "P-002",
                  name: "FROST_MAGE",
                  status: "pending",
                  game: "MLBB",
                },
                {
                  id: "P-003",
                  name: "BLADE_MASTER",
                  status: "analyzed",
                  game: "MLBB",
                },
                {
                  id: "P-004",
                  name: "MYSTIC_ARCHER",
                  status: "flagged",
                  game: "MLBB",
                },
              ].map((player) => (
                <div
                  key={player.id}
                  className="flex items-center justify-between p-2 bg-neutral-800 rounded hover:bg-neutral-700 transition-colors cursor-pointer"
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`w-2 h-2 rounded-full ${
                        player.status === "analyzed"
                          ? "bg-white"
                          : player.status === "pending"
                          ? "bg-orange-500"
                          : "bg-red-500"
                      }`}
                    ></div>
                    <div>
                      <div className="text-xs text-white font-mono">
                        {player.id}
                      </div>
                      <div className="text-xs text-neutral-500">
                        {player.name}
                      </div>
                    </div>
                  </div>
                  <div className="text-xs text-neutral-400">{player.game}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Feedback Activity Log */}
        <Card className="lg:col-span-4 bg-neutral-900 border-neutral-700">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-neutral-300 tracking-wider">
              FEEDBACK ACTIVITY
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {[
                {
                  time: "17/01/2025 14:29",
                  player: "SHADOW_STRIKER",
                  action: "received feedback for",
                  type: "Screenshot Analysis",
                  model: "MVP Detection",
                },
                {
                  time: "17/01/2025 14:12",
                  player: "FROST_MAGE",
                  action: "uploaded video for",
                  type: "Team Sync Analysis",
                  model: "Behavioral Model",
                },
                {
                  time: "17/01/2025 13:55",
                  player: "BLADE_MASTER",
                  action: "completed analysis of",
                  type: "Performance Review",
                  model: "EasyOCR + LLM",
                },
                {
                  time: "17/01/2025 13:33",
                  player: "MYSTIC_ARCHER",
                  action: "flagged for review in",
                  type: "Synergy Analysis",
                  model: "Team Behavior",
                },
                {
                  time: "17/01/2025 13:15",
                  player: "STORM_RIDER",
                  action: "received coaching for",
                  type: "Medal Detection",
                  model: "YOLO + OCR",
                },
              ].map((log, index) => (
                <div
                  key={index}
                  className="text-xs border-l-2 border-orange-500 pl-3 hover:bg-neutral-800 p-2 rounded transition-colors"
                >
                  <div className="text-neutral-500 font-mono">{log.time}</div>
                  <div className="text-white">
                    Player{" "}
                    <span className="text-orange-500 font-mono">
                      {log.player}
                    </span>{" "}
                    {log.action}{" "}
                    <span className="text-white font-mono">{log.type}</span>
                    {log.model && (
                      <span>
                        {" "}
                        via{" "}
                        <span className="text-orange-500 font-mono">
                          {log.model}
                        </span>
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Coaching Feed Memory */}
        <Card className="lg:col-span-4 bg-neutral-900 border-neutral-700">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-neutral-300 tracking-wider">
              COACHING FEED MEMORY
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col items-center">
            {/* Wireframe Sphere */}
            <div className="relative w-32 h-32 mb-4">
              <div className="absolute inset-0 border-2 border-white rounded-full opacity-60 animate-pulse"></div>
              <div className="absolute inset-2 border border-white rounded-full opacity-40"></div>
              <div className="absolute inset-4 border border-white rounded-full opacity-20"></div>
              {/* Grid lines */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-full h-px bg-white opacity-30"></div>
              </div>
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-px h-full bg-white opacity-30"></div>
              </div>
            </div>

            <div className="text-xs text-neutral-500 space-y-1 w-full font-mono">
              <div className="flex justify-between">
                <span># 2025-01-17 14:30 UTC</span>
              </div>
              <div className="text-white">
                {"> [AI:coach_v2] ::: ANALYSIS >> ^^^ processing player data"}
              </div>
              <div className="text-orange-500">
                {"> TRACE | 1847.3092847.891...xR7"}
              </div>
              <div className="text-white">{"> MODEL READY"}</div>
              <div className="text-neutral-400">
                {
                  '> FEEDBACK >> "...player improvement detected... synergy analysis complete"'
                }
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Analysis Usage Trends */}
        <Card className="lg:col-span-8 bg-neutral-900 border-neutral-700">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-neutral-300 tracking-wider">
              ANALYSIS USAGE TRENDS
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-48 relative">
              {/* Chart Grid */}
              <div className="absolute inset-0 grid grid-cols-8 grid-rows-6 opacity-20">
                {Array.from({ length: 48 }).map((_, i) => (
                  <div key={i} className="border border-neutral-700"></div>
                ))}
              </div>

              {/* Chart Lines */}
              <svg className="absolute inset-0 w-full h-full">
                {/* Screenshots line */}
                <polyline
                  points="0,140 50,120 100,130 150,110 200,115 250,105 300,120 350,100"
                  fill="none"
                  stroke="#f97316"
                  strokeWidth="2"
                />
                {/* Videos line */}
                <polyline
                  points="0,160 50,155 100,150 150,145 200,150 250,155 300,145 350,140"
                  fill="none"
                  stroke="#ffffff"
                  strokeWidth="2"
                  strokeDasharray="5,5"
                />
              </svg>

              {/* Y-axis labels */}
              <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-neutral-500 -ml-5 font-mono">
                <span>200</span>
                <span>150</span>
                <span>100</span>
                <span>50</span>
              </div>

              {/* X-axis labels */}
              <div className="absolute bottom-0 left-0 w-full flex justify-between text-xs text-neutral-500 -mb-6 font-mono">
                <span>Jan 10, 2025</span>
                <span>Jan 17, 2025</span>
              </div>
            </div>

            {/* Legend */}
            <div className="flex justify-center gap-6 mt-4">
              <div className="flex items-center gap-2">
                <div className="w-4 h-px bg-orange-500"></div>
                <span className="text-xs text-neutral-400">Screenshots</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-px bg-white border-dashed border-b"></div>
                <span className="text-xs text-neutral-400">Videos</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Model Accuracy Overview */}
        <Card className="lg:col-span-4 bg-neutral-900 border-neutral-700">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-neutral-300 tracking-wider">
              MODEL ACCURACY OVERVIEW
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 gap-4">
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                  <span className="text-xs text-white font-medium">
                    Detection Modules
                  </span>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-neutral-400">MVP Detection</span>
                    <span className="text-white font-bold font-mono">
                      94.7%
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-neutral-400">Medal Recognition</span>
                    <span className="text-white font-bold font-mono">
                      89.2%
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-neutral-400">Behavior Analysis</span>
                    <span className="text-white font-bold font-mono">
                      87.5%
                    </span>
                  </div>
                </div>
              </div>

              <div>
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                  <span className="text-xs text-red-500 font-medium">
                    Error Rates
                  </span>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-neutral-400">OCR Timeout</span>
                    <span className="text-white font-bold font-mono">2.1%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-neutral-400">Low Confidence</span>
                    <span className="text-white font-bold font-mono">4.3%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-neutral-400">Processing Failed</span>
                    <span className="text-white font-bold font-mono">1.2%</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
