"use client";

import { useState } from "react";
import {
  ChevronRight,
  Monitor,
  Settings,
  Shield,
  Target,
  Users,
  Bell,
  RefreshCw,
  Camera,
  TrendingUp,
  Brain,
  Video,
  Play,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import ClientOnly from "@/components/ClientOnly";
import PlayerHubPage from "./player-hub/page";
import PlayerNetworkPage from "./player-network/page";
import ScreenshotAnalysisPage from "./screenshot-analysis/page";
import AnalysisOpsPage from "./analysis-ops/page";
import PlayerInsightsPage from "./player-insights/page";
import AiCoachStatusPage from "./ai-coach-status/page";
import VideoAnalysisPage from "./video-analysis/page";

export default function TacticalDashboard() {
  const [activeSection, setActiveSection] = useState("overview");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div
        className={`${
          sidebarCollapsed ? "w-16" : "w-70"
        } bg-neutral-900 border-r border-neutral-700 transition-all duration-300 fixed md:relative z-50 md:z-auto h-full md:h-auto ${
          !sidebarCollapsed ? "md:block" : ""
        }`}
      >
        <div className="p-4">
          <div className="flex items-center justify-between mb-8">
            <div className={`${sidebarCollapsed ? "hidden" : "block"}`}>
              <h1 className="text-orange-500 font-bold text-lg tracking-wider">
                AI COACH
              </h1>
              <p className="text-neutral-500 text-xs">v2.1.7 ESPORTS</p>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="text-neutral-400 hover:text-orange-500"
            >
              <ChevronRight
                className={`w-4 h-4 sm:w-5 sm:h-5 transition-transform ${
                  sidebarCollapsed ? "" : "rotate-180"
                }`}
              />
            </Button>
          </div>

          <nav className="space-y-2">
            {[
              { id: "overview", icon: Monitor, label: "PLAYER HUB" },
              { id: "players", icon: Users, label: "PLAYER NETWORK" },
              {
                id: "screenshot-analysis",
                icon: Camera,
                label: "SCREENSHOT ANALYSIS",
              },
              { id: "video-analysis", icon: Video, label: "VIDEO ANALYSIS" },
              { id: "operations", icon: Target, label: "ANALYSIS OPS" },
              { id: "insights", icon: Brain, label: "PLAYER INSIGHTS" },
              { id: "systems", icon: Settings, label: "AI COACH STATUS" },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveSection(item.id)}
                className={`w-full flex items-center gap-3 p-3 rounded transition-colors ${
                  activeSection === item.id
                    ? "bg-orange-500 text-white"
                    : "text-neutral-400 hover:text-white hover:bg-neutral-800"
                }`}
              >
                <item.icon className="w-5 h-5 md:w-5 md:h-5 sm:w-6 sm:h-6" />
                {!sidebarCollapsed && (
                  <span className="text-sm font-medium">{item.label}</span>
                )}
              </button>
            ))}
          </nav>

          {!sidebarCollapsed && (
            <div className="mt-8 p-4 bg-neutral-800 border border-neutral-700 rounded">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                <span className="text-xs text-white">SYSTEM ONLINE</span>
              </div>
              <div className="text-xs text-neutral-500">
                <ClientOnly
                  fallback={
                    <>
                      <div>UPTIME: --:--:--</div>
                      <div>PLAYERS: --- ACTIVE</div>
                      <div>ANALYSES: -- RUNNING</div>
                    </>
                  }
                >
                  <div>UPTIME: 72:14:33</div>
                  <div>PLAYERS: 1,247 ACTIVE</div>
                  <div>ANALYSES: 89 RUNNING</div>
                </ClientOnly>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Mobile Overlay */}
      {!sidebarCollapsed && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={() => setSidebarCollapsed(true)}
        />
      )}

      {/* Main Content */}
      <div
        className={`flex-1 flex flex-col ${!sidebarCollapsed ? "md:ml-0" : ""}`}
      >
        {/* Top Toolbar */}
        <div className="h-16 bg-neutral-800 border-b border-neutral-700 flex items-center justify-between px-6">
          <div className="flex items-center gap-4">
            <div className="text-sm text-neutral-400">
              AI COACH DASHBOARD /{" "}
              <span className="text-orange-500">
                {activeSection === "overview" && "PLAYER HUB"}
                {activeSection === "players" && "PLAYER NETWORK"}
                {activeSection === "upload" && "UPLOAD CENTER"}
                {activeSection === "operations" && "ANALYSIS OPS"}
                {activeSection === "insights" && "PLAYER INSIGHTS"}
                {activeSection === "systems" && "AI COACH STATUS"}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-xs text-neutral-500">
              <ClientOnly fallback="LAST UPDATE: --/--/---- --:-- UTC">
                LAST UPDATE: 17/01/2025 14:30 UTC
              </ClientOnly>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="text-neutral-400 hover:text-orange-500"
            >
              <Bell className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="text-neutral-400 hover:text-orange-500"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Dashboard Content */}
        <div className="flex-1 overflow-auto">
          {activeSection === "overview" && <PlayerHubPage />}
          {activeSection === "players" && <PlayerNetworkPage />}
          {activeSection === "screenshot-analysis" && (
            <ScreenshotAnalysisPage />
          )}
          {activeSection === "video-analysis" && <VideoAnalysisPage />}
          {activeSection === "operations" && <AnalysisOpsPage />}
          {activeSection === "insights" && <PlayerInsightsPage />}
          {activeSection === "systems" && <AiCoachStatusPage />}
        </div>
      </div>
    </div>
  );
}
