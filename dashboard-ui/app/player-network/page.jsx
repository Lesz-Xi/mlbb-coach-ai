"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Search,
  Filter,
  MoreHorizontal,
  MapPin,
  Clock,
  Shield,
  Trophy,
  Target,
  Users,
} from "lucide-react";

export default function PlayerNetworkPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedPlayer, setSelectedPlayer] = useState(null);

  const players = [
    {
      id: "P-001",
      name: "SHADOW_STRIKER",
      status: "analyzed",
      gameMode: "Ranked",
      lastSeen: "2 min ago",
      matches: 47,
      performance: "high",
      role: "Assassin",
      rank: "Mythic III",
    },
    {
      id: "P-002",
      name: "FROST_MAGE",
      status: "pending",
      gameMode: "Classic",
      lastSeen: "15 min ago",
      matches: 32,
      performance: "medium",
      role: "Mage",
      rank: "Legend I",
    },
    {
      id: "P-003",
      name: "BLADE_MASTER",
      status: "analyzed",
      gameMode: "Mythic+",
      lastSeen: "1 min ago",
      matches: 63,
      performance: "high",
      role: "Fighter",
      rank: "Mythic V",
    },
    {
      id: "P-004",
      name: "MYSTIC_ARCHER",
      status: "flagged",
      gameMode: "Ranked",
      lastSeen: "3 hours ago",
      matches: 28,
      performance: "critical",
      role: "Marksman",
      rank: "Epic I",
    },
    {
      id: "P-005",
      name: "STORM_RIDER",
      status: "analyzed",
      gameMode: "Classic",
      lastSeen: "5 min ago",
      matches: 41,
      performance: "medium",
      role: "Mage",
      rank: "Legend II",
    },
    {
      id: "P-006",
      name: "IRON_GUARDIAN",
      status: "training",
      gameMode: "AI Practice",
      lastSeen: "1 day ago",
      matches: 12,
      performance: "low",
      role: "Tank",
      rank: "Master IV",
    },
    {
      id: "P-007",
      name: "VOID_HUNTER",
      status: "analyzed",
      gameMode: "Ranked",
      lastSeen: "8 min ago",
      matches: 55,
      performance: "high",
      role: "Assassin",
      rank: "Mythic II",
    },
    {
      id: "P-008",
      name: "LIGHT_BEARER",
      status: "pending",
      gameMode: "Classic",
      lastSeen: "22 min ago",
      matches: 38,
      performance: "medium",
      role: "Support",
      rank: "Legend III",
    },
  ];

  const filteredPlayers = players.filter(
    (player) =>
      player.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      player.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      player.role.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getStatusColor = (status) => {
    switch (status) {
      case "analyzed":
        return "bg-white/20 text-white";
      case "pending":
        return "bg-orange-500/20 text-orange-500";
      case "training":
        return "bg-orange-500/20 text-orange-500";
      case "flagged":
        return "bg-red-500/20 text-red-500";
      default:
        return "bg-neutral-500/20 text-neutral-300";
    }
  };

  const getPerformanceColor = (performance) => {
    switch (performance) {
      case "critical":
        return "bg-red-500/20 text-red-500";
      case "high":
        return "bg-white/20 text-white";
      case "medium":
        return "bg-orange-500/20 text-orange-500";
      case "low":
        return "bg-neutral-500/20 text-neutral-300";
      default:
        return "bg-neutral-500/20 text-neutral-300";
    }
  };

  const getRoleIcon = (role) => {
    switch (role) {
      case "Assassin":
        return <Target className="w-4 h-4" />;
      case "Tank":
        return <Shield className="w-4 h-4" />;
      case "Support":
        return <Users className="w-4 h-4" />;
      case "Marksman":
        return <Trophy className="w-4 h-4" />;
      default:
        return <Target className="w-4 h-4" />;
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-wider">
            PLAYER NETWORK
          </h1>
          <p className="text-sm text-neutral-400">
            Monitor and manage registered players
          </p>
        </div>
        <div className="flex gap-2">
          <Button className="bg-orange-500 hover:bg-orange-600 text-white">
            Add Player
          </Button>
          <Button className="bg-orange-500 hover:bg-orange-600 text-white">
            <Filter className="w-4 h-4 mr-2" />
            Filter
          </Button>
        </div>
      </div>

      {/* Search and Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        <Card className="lg:col-span-1 bg-neutral-900 border-neutral-700">
          <CardContent className="p-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-neutral-400" />
              <Input
                placeholder="Search players..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 bg-neutral-800 border-neutral-600 text-white placeholder-neutral-400"
              />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-neutral-900 border-neutral-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-neutral-400 tracking-wider">
                  ACTIVE PLAYERS
                </p>
                <p className="text-2xl font-bold text-white font-mono">1,247</p>
              </div>
              <Shield className="w-8 h-8 text-white" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-neutral-900 border-neutral-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-neutral-400 tracking-wider">
                  FLAGGED
                </p>
                <p className="text-2xl font-bold text-red-500 font-mono">23</p>
              </div>
              <Shield className="w-8 h-8 text-red-500" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-neutral-900 border-neutral-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-neutral-400 tracking-wider">
                  IN TRAINING
                </p>
                <p className="text-2xl font-bold text-orange-500 font-mono">
                  156
                </p>
              </div>
              <Shield className="w-8 h-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Player List */}
      <Card className="bg-neutral-900 border-neutral-700">
        <CardHeader>
          <CardTitle className="text-sm font-medium text-neutral-300 tracking-wider">
            PLAYER ROSTER
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-neutral-700">
                  <th className="text-left py-3 px-4 text-xs font-medium text-neutral-400 tracking-wider">
                    PLAYER ID
                  </th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-neutral-400 tracking-wider">
                    USERNAME
                  </th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-neutral-400 tracking-wider">
                    STATUS
                  </th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-neutral-400 tracking-wider">
                    GAME MODE
                  </th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-neutral-400 tracking-wider">
                    ROLE
                  </th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-neutral-400 tracking-wider">
                    RANK
                  </th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-neutral-400 tracking-wider">
                    MATCHES
                  </th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-neutral-400 tracking-wider">
                    PERFORMANCE
                  </th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-neutral-400 tracking-wider">
                    ACTIONS
                  </th>
                </tr>
              </thead>
              <tbody>
                {filteredPlayers.map((player, index) => (
                  <tr
                    key={player.id}
                    className={`border-b border-neutral-800 hover:bg-neutral-800 transition-colors cursor-pointer ${
                      index % 2 === 0 ? "bg-neutral-900" : "bg-neutral-850"
                    }`}
                    onClick={() => setSelectedPlayer(player)}
                  >
                    <td className="py-3 px-4 text-sm text-white font-mono">
                      {player.id}
                    </td>
                    <td className="py-3 px-4 text-sm text-white">
                      {player.name}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div
                          className={`w-2 h-2 rounded-full ${
                            player.status === "analyzed"
                              ? "bg-white"
                              : player.status === "pending"
                              ? "bg-orange-500"
                              : player.status === "training"
                              ? "bg-orange-500"
                              : "bg-red-500"
                          }`}
                        ></div>
                        <span className="text-xs text-neutral-300 uppercase tracking-wider">
                          {player.status}
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <MapPin className="w-3 h-3 text-neutral-400" />
                        <span className="text-sm text-neutral-300">
                          {player.gameMode}
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        {getRoleIcon(player.role)}
                        <span className="text-sm text-neutral-300">
                          {player.role}
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-sm text-white font-mono">
                      {player.rank}
                    </td>
                    <td className="py-3 px-4 text-sm text-white font-mono">
                      {player.matches}
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`text-xs px-2 py-1 rounded uppercase tracking-wider ${getPerformanceColor(
                          player.performance
                        )}`}
                      >
                        {player.performance}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="text-neutral-400 hover:text-orange-500"
                      >
                        <MoreHorizontal className="w-4 h-4" />
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Player Detail Modal */}
      {selectedPlayer && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <Card className="bg-neutral-900 border-neutral-700 w-full max-w-2xl">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-lg font-bold text-white tracking-wider">
                  {selectedPlayer.name}
                </CardTitle>
                <p className="text-sm text-neutral-400 font-mono">
                  {selectedPlayer.id}
                </p>
              </div>
              <Button
                variant="ghost"
                onClick={() => setSelectedPlayer(null)}
                className="text-neutral-400 hover:text-white"
              >
                ✕
              </Button>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-neutral-400 tracking-wider mb-1">
                    STATUS
                  </p>
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-2 h-2 rounded-full ${
                        selectedPlayer.status === "analyzed"
                          ? "bg-white"
                          : selectedPlayer.status === "pending"
                          ? "bg-orange-500"
                          : selectedPlayer.status === "training"
                          ? "bg-orange-500"
                          : "bg-red-500"
                      }`}
                    ></div>
                    <span className="text-sm text-white uppercase tracking-wider">
                      {selectedPlayer.status}
                    </span>
                  </div>
                </div>
                <div>
                  <p className="text-xs text-neutral-400 tracking-wider mb-1">
                    GAME MODE
                  </p>
                  <p className="text-sm text-white">
                    {selectedPlayer.gameMode}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-neutral-400 tracking-wider mb-1">
                    ROLE
                  </p>
                  <div className="flex items-center gap-2">
                    {getRoleIcon(selectedPlayer.role)}
                    <span className="text-sm text-white">
                      {selectedPlayer.role}
                    </span>
                  </div>
                </div>
                <div>
                  <p className="text-xs text-neutral-400 tracking-wider mb-1">
                    RANK
                  </p>
                  <p className="text-sm text-white font-mono">
                    {selectedPlayer.rank}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-neutral-400 tracking-wider mb-1">
                    MATCHES ANALYZED
                  </p>
                  <p className="text-sm text-white font-mono">
                    {selectedPlayer.matches}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-neutral-400 tracking-wider mb-1">
                    PERFORMANCE TIER
                  </p>
                  <span
                    className={`text-xs px-2 py-1 rounded uppercase tracking-wider ${getPerformanceColor(
                      selectedPlayer.performance
                    )}`}
                  >
                    {selectedPlayer.performance}
                  </span>
                </div>
              </div>
              <div className="flex gap-2 pt-4">
                <Button className="bg-orange-500 hover:bg-orange-600 text-white">
                  View Analysis
                </Button>
                <Button
                  variant="outline"
                  className="border-neutral-700 text-neutral-400 hover:bg-neutral-800 hover:text-neutral-300 bg-transparent"
                >
                  Send Coaching
                </Button>
                <Button
                  variant="outline"
                  className="border-neutral-700 text-neutral-400 hover:bg-neutral-800 hover:text-neutral-300 bg-transparent"
                >
                  Flag Player
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
