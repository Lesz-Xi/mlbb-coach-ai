import { NextResponse } from "next/server";
import fetch from "node-fetch";

// Configuration for the skillshift-ai backend
const SKILLSHIFT_AI_BASE_URL =
  process.env.SKILLSHIFT_AI_URL || "http://localhost:8000";

export async function GET() {
  try {
    // Check backend health with multiple fallback strategies
    let healthResponse;
    
    // First, try a quick health check with shorter timeout
    try {
      healthResponse = await fetch(
        `${SKILLSHIFT_AI_BASE_URL}/api/health`,
        {
          method: "GET",
          signal: AbortSignal.timeout(8000), // 8 second timeout
        }
      );
    } catch (quickError) {
      // If quick check fails, try the basic health endpoint
      console.log("Quick health check failed, trying basic endpoint...");
      healthResponse = await fetch(
        `${SKILLSHIFT_AI_BASE_URL}/health`,
        {
          method: "GET",
          signal: AbortSignal.timeout(5000), // 5 second timeout for basic
        }
      );
    }

    if (!healthResponse.ok) {
      console.error("Backend health check failed:", healthResponse.status);
      return NextResponse.json(
        {
          status: "degraded",
          frontend: "healthy",
          backend: "degraded",
          backend_status: healthResponse.status,
          error: "Backend health check failed",
          timestamp: new Date().toISOString(),
        },
        { status: 503 }
      );
    }

    const backendHealth = await healthResponse.json();

    // Return combined health status
    return NextResponse.json(
      {
        status: "healthy",
        frontend: "healthy",
        backend: "healthy",
        backend_info: backendHealth,
        services: {
          frontend: "Next.js Dashboard",
          backend: "SkillShift AI",
          database: backendHealth.hero_database_size > 0 ? "connected" : "disconnected",
        },
        timestamp: new Date().toISOString(),
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Health check error:", error);

    // Determine error type
    let errorType = "unknown";
    let errorMessage = error.message;

    if (error.name === "AbortError" || error.message.includes("timeout")) {
      errorType = "timeout";
      errorMessage = "Backend connection timeout";
    } else if (error.code === "ECONNREFUSED") {
      errorType = "connection_refused";
      errorMessage = "Backend is not running";
    } else if (error.code === "ENOTFOUND") {
      errorType = "dns_error";
      errorMessage = "Backend host not found";
    }

    return NextResponse.json(
      {
        status: "degraded",
        frontend: "healthy",
        backend: "offline",
        error: errorMessage,
        error_type: errorType,
        backend_url: SKILLSHIFT_AI_BASE_URL,
        timestamp: new Date().toISOString(),
        troubleshooting: {
          suggestions: [
            "Check if backend is running on port 8000",
            "Verify backend health endpoint: curl http://localhost:8000/api/health",
            "Check network connectivity",
            "Review backend logs for errors",
          ],
        },
      },
      { status: 503 }
    );
  }
}