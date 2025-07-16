import { uploadFile, validateFile, formatFileSize } from '@/lib/local-storage';
import { NextResponse } from 'next/server';

export async function POST(request) {
  try {
    const formData = await request.formData();
    const file = formData.get('file');
    const analysisType = formData.get('analysisType') || 'general';
    
    if (!file) {
      return NextResponse.json(
        { success: false, error: 'No file provided' },
        { status: 400 }
      );
    }

    // Validate file
    const validation = validateFile(file);
    if (!validation.valid) {
      return NextResponse.json(
        { success: false, error: validation.error },
        { status: 400 }
      );
    }

    // Upload to local storage
    const uploadResult = await uploadFile(file, file.name);
    
    if (!uploadResult.success) {
      return NextResponse.json(
        { success: false, error: uploadResult.error },
        { status: 500 }
      );
    }

    // Create file metadata (in a real app, you'd save this to database)
    const fileMetadata = {
      id: Date.now(), // In real app, use UUID from database
      filename: uploadResult.key,
      originalName: file.name,
      fileType: file.type,
      fileSize: file.size,
      formattedSize: formatFileSize(file.size),
      localKey: uploadResult.key,
      localPath: uploadResult.path,
      fileUrl: uploadResult.url,
      location: uploadResult.location,
      status: 'uploaded',
      analysisType,
      uploadDate: new Date().toISOString(),
    };

    return NextResponse.json({
      success: true,
      message: 'File uploaded successfully',
      file: fileMetadata,
    });

  } catch (error) {
    console.error('Upload API error:', error);
    return NextResponse.json(
      { success: false, error: 'Upload failed' },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    message: 'Upload endpoint is working',
    supportedMethods: ['POST'],
  });
}