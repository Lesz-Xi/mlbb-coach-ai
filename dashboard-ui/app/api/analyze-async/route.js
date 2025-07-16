import { NextResponse } from "next/server";
import FormData from "form-data";
import fetch from "node-fetch";

// Configuration for the skillshift-ai backend
const SKILLSHIFT_AI_BASE_URL =
  process.env.SKILLSHIFT_AI_URL || "http://localhost:8000";
const DEFAULT_IGN = process.env.DEFAULT_IGN || "Lesz XVII";

export async function POST(request) {
  try {
    const formData = await request.formData();
    const file = formData.get("file");
    const ign = formData.get("ign") || DEFAULT_IGN;
    const hero_override = formData.get("hero_override") || "";

    if (!file) {
      return NextResponse.json(
        { success: false, error: "No file provided for analysis" },
        { status: 400 }
      );
    }

    // Validate file type - should be image
    if (!file.type.startsWith("image/")) {
      return NextResponse.json(
        {
          success: false,
          error: "Only image files are supported for analysis",
        },
        { status: 400 }
      );
    }

    // Health check before processing
    try {
      const healthResponse = await fetch(
        `${SKILLSHIFT_AI_BASE_URL}/api/health-isolated`,
        {
          method: "GET",
          signal: AbortSignal.timeout(2000),
        }
      );

      if (!healthResponse.ok) {
        return NextResponse.json(
          {
            success: false,
            error: "Analysis service is currently unavailable",
            details: "Backend health check failed",
          },
          { status: 503 }
        );
      }
    } catch (healthError) {
      return NextResponse.json(
        {
          success: false,
          error: "Cannot connect to analysis service",
          details: "Please ensure the backend is running on port 8000",
        },
        { status: 503 }
      );
    }

    // Create FormData for skillshift-ai backend
    const backendFormData = new FormData();
    const fileBuffer = Buffer.from(await file.arrayBuffer());
    backendFormData.append("file", fileBuffer, {
      filename: file.name,
      contentType: file.type,
    });

    // Call the new async job creation endpoint
    const jobResponse = await fetch(
      `${SKILLSHIFT_AI_BASE_URL}/api/jobs?ign=${encodeURIComponent(ign)}&hero_override=${encodeURIComponent(hero_override)}`,
      {
        method: "POST",
        body: backendFormData,
        headers: backendFormData.getHeaders(),
        signal: AbortSignal.timeout(10000), // 10 second timeout for job creation
      }
    );

    if (!jobResponse.ok) {
      const errorText = await jobResponse.text();
      return NextResponse.json(
        {
          success: false,
          error: "Failed to create analysis job",
          details: errorText,
        },
        { status: jobResponse.status }
      );
    }

    const jobResult = await jobResponse.json();

    // Return job information to frontend for polling
    return NextResponse.json({
      success: true,
      job_id: jobResult.job_id,
      state: jobResult.state,
      message: jobResult.message || "Analysis job created successfully",
      estimated_completion: jobResult.estimated_completion || "2-5 minutes",
      status_url: `/api/job-status/${jobResult.job_id}`,
      metadata: {
        ign: ign,
        filename: file.name,
        fileSize: file.size,
        timestamp: new Date().toISOString(),
        backend_url: SKILLSHIFT_AI_BASE_URL,
      },
    });

  } catch (error) {
    console.error("Async analysis API error:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to create analysis job",
        details: error.message,
      },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    message: "MLBB Coach AI Async Analysis Endpoint",
    status: "operational",
    backend_url: SKILLSHIFT_AI_BASE_URL,
    supported_methods: ["POST"],
    required_fields: ["file"],
    optional_fields: ["ign", "hero_override"],
    description: "Creates analysis jobs and returns job ID for polling",
  });
} 