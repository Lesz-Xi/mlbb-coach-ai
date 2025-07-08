import { useCallback, useState } from 'react'

const FileUpload = ({ onFileUpload, isLoading }) => {
  const [isDragOver, setIsDragOver] = useState(false)

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
    
    const files = e.dataTransfer.files
    if (files.length > 0) {
      const file = files[0]
      if (file.type.startsWith('image/')) {
        onFileUpload(file)
      }
    }
  }, [onFileUpload])

  const handleFileSelect = useCallback((e) => {
    const file = e.target.files?.[0]
    if (file) {
      onFileUpload(file)
    }
  }, [onFileUpload])

  return (
    <div className="bg-gray-800 rounded-lg p-8 shadow-xl">
      <h2 className="text-2xl font-bold text-mlbb-gold mb-6 text-center">
        Upload Match Screenshot
      </h2>
      
      <div
        className={`border-2 border-dashed rounded-lg p-12 text-center transition-all ${
          isDragOver
            ? 'border-mlbb-gold bg-mlbb-gold/10'
            : 'border-gray-600 hover:border-mlbb-gold/50'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {isLoading ? (
          <div className="space-y-4">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-mlbb-gold mx-auto"></div>
            <p className="text-gray-300">Analyzing your match...</p>
          </div>
        ) : (
          <>
            <div className="space-y-4">
              <svg
                className="mx-auto h-12 w-12 text-gray-400"
                stroke="currentColor"
                fill="none"
                viewBox="0 0 48 48"
              >
                <path
                  d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <div>
                <p className="text-gray-300 text-lg mb-2">
                  Drag and drop your match screenshot here
                </p>
                <p className="text-gray-500 text-sm mb-4">
                  or click to select a file
                </p>
                <label className="cursor-pointer">
                  <span className="inline-block bg-mlbb-gold text-mlbb-dark font-semibold py-2 px-6 rounded-lg hover:bg-yellow-400 transition-colors">
                    Choose File
                  </span>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </label>
              </div>
            </div>
            <div className="mt-6 text-sm text-gray-500">
              <p>Supported formats: PNG, JPG, JPEG</p>
              <p>Max file size: 10MB</p>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

export default FileUpload