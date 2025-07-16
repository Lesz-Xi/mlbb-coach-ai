import { promises as fs } from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';

const UPLOAD_DIR = path.join(process.cwd(), 'uploads');

export const ensureUploadDir = async () => {
  try {
    await fs.access(UPLOAD_DIR);
  } catch {
    await fs.mkdir(UPLOAD_DIR, { recursive: true });
  }
};

export const uploadFile = async (file, originalFilename) => {
  try {
    await ensureUploadDir();
    
    const fileExtension = originalFilename.split('.').pop();
    const uniqueFilename = `${uuidv4()}.${fileExtension}`;
    const filePath = path.join(UPLOAD_DIR, uniqueFilename);
    
    // Convert file to buffer
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    
    // Write file to local storage
    await fs.writeFile(filePath, buffer);
    
    return {
      success: true,
      key: uniqueFilename,
      path: filePath,
      url: `/api/files/${uniqueFilename}`,
      location: `http://localhost:3000/api/files/${uniqueFilename}`,
    };
  } catch (error) {
    console.error('Local upload error:', error);
    return {
      success: false,
      error: error.message,
    };
  }
};

export const getFileUrl = async (key) => {
  try {
    const filePath = path.join(UPLOAD_DIR, key);
    await fs.access(filePath);
    return { 
      success: true, 
      url: `/api/files/${key}` 
    };
  } catch (error) {
    return { 
      success: false, 
      error: 'File not found' 
    };
  }
};

export const deleteFile = async (key) => {
  try {
    const filePath = path.join(UPLOAD_DIR, key);
    await fs.unlink(filePath);
    return { success: true };
  } catch (error) {
    return { 
      success: false, 
      error: error.message 
    };
  }
};

export const validateFile = (file) => {
  const allowedTypes = [
    'image/png', 
    'image/jpeg', 
    'image/jpg',
    'video/mp4', 
    'video/quicktime',
    'video/webm'
  ];
  
  const maxSize = 100 * 1024 * 1024; // 100MB
  
  if (!allowedTypes.includes(file.type)) {
    return { 
      valid: false, 
      error: `Invalid file type. Allowed: ${allowedTypes.join(', ')}` 
    };
  }
  
  if (file.size > maxSize) {
    return { 
      valid: false, 
      error: `File too large. Maximum size: ${maxSize / (1024 * 1024)}MB` 
    };
  }
  
  return { valid: true };
};

export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};