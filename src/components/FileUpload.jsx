import { useCallback, useState, useRef } from "react";

const FileUpload = ({
  onFileUpload,
  isLoading,
  uploadedScreenshots = [],
  useEnhanced = true,
  onRemoveFile = null,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragOver(false);

      const files = Array.from(e.dataTransfer.files);
      const imageFiles = files.filter((file) => file.type.startsWith("image/"));

      if (imageFiles.length > 0) {
        imageFiles.forEach((file) => onFileUpload(file));
      }
    },
    [onFileUpload]
  );

  const handleFileSelect = useCallback(
    (e) => {
      const files = Array.from(e.target.files || []);
      const imageFiles = files.filter((file) => file.type.startsWith("image/"));

      console.log(
        `Selected ${files.length} files, ${imageFiles.length} are images`
      );

      if (imageFiles.length > 0) {
        imageFiles.forEach((file) => onFileUpload(file));
      }

      // Reset input to allow selecting the same files again if needed
      e.target.value = "";
    },
    [onFileUpload]
  );

  const handleChooseFiles = useCallback(() => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }, []);

  const detectScreenshotType = (filename) => {
    const name = filename.toLowerCase();
    if (
      name.includes("score") ||
      name.includes("kda") ||
      name.includes("match")
    ) {
      return "Scoreboard";
    } else if (
      name.includes("summary") ||
      name.includes("result") ||
      name.includes("timeline")
    ) {
      return "Match Summary";
    } else if (name.includes("damage") || name.includes("dmg")) {
      return "Damage Report";
    }
    return "Unknown Type";
  };

  const shouldShowUpload = !useEnhanced || uploadedScreenshots.length < 2;
  const canAnalyze = useEnhanced
    ? uploadedScreenshots.length >= 2
    : uploadedScreenshots.length >= 1;

  return (
    <div className="bg-gray-800 rounded-lg p-8 shadow-xl">
      <h2 className="text-2xl font-bold text-mlbb-gold mb-6 text-center">
        Upload Match Screenshots
      </h2>

      {/* Display uploaded files */}
      {uploadedScreenshots.length > 0 && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-200 mb-3">
            Uploaded Screenshots ({uploadedScreenshots.length})
          </h3>
          <div className="space-y-2">
            {uploadedScreenshots.map((screenshot, index) => (
              <div
                key={index}
                className="flex items-center justify-between bg-gray-700 p-3 rounded-lg"
              >
                <div className="flex items-center space-x-3">
                  <div className="bg-mlbb-gold text-mlbb-dark px-2 py-1 rounded text-sm font-bold">
                    {index + 1}
                  </div>
                  <div>
                    <p className="text-gray-200 text-sm truncate max-w-40">
                      {screenshot.name}
                    </p>
                    <p className="text-gray-400 text-xs">
                      {detectScreenshotType(screenshot.name)}
                    </p>
                  </div>
                </div>
                {onRemoveFile && (
                  <button
                    onClick={() => onRemoveFile(index)}
                    className="text-red-400 hover:text-red-300 transition-colors"
                    title="Remove screenshot"
                  >
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M6 18L18 6M6 6l12 12"
                      />
                    </svg>
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Upload area */}
      {shouldShowUpload && (
        <div
          className={`border-2 border-dashed rounded-lg p-12 text-center transition-all ${
            isDragOver
              ? "border-mlbb-gold bg-mlbb-gold/10"
              : "border-gray-600 hover:border-mlbb-gold/50"
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
                    {useEnhanced && uploadedScreenshots.length === 0
                      ? "Upload screenshot 1 of 2 (Scoreboard recommended)"
                      : useEnhanced && uploadedScreenshots.length === 1
                      ? "Upload screenshot 2 of 2 (Match Summary recommended)"
                      : "Drag and drop your match screenshots here"}
                  </p>
                  <p className="text-gray-500 text-sm mb-4">
                    or click to select files
                  </p>

                  {/* Enhanced file selection buttons */}
                  <div className="space-y-3">
                    <button
                      onClick={handleChooseFiles}
                      className="inline-block bg-mlbb-gold text-mlbb-dark font-semibold py-2 px-6 rounded-lg hover:bg-yellow-400 transition-colors"
                    >
                      Choose Files
                    </button>

                    {/* Instructions for multiple selection */}
                    <div className="text-xs text-gray-400 space-y-1">
                      <p>
                        💡 <strong>Multiple file selection tips:</strong>
                      </p>
                      <p>
                        • Hold{" "}
                        <kbd className="bg-gray-600 px-1 rounded">Ctrl</kbd>{" "}
                        (Windows) or{" "}
                        <kbd className="bg-gray-600 px-1 rounded">Cmd</kbd>{" "}
                        (Mac) to select multiple files
                      </p>
                      <p>
                        • Hold{" "}
                        <kbd className="bg-gray-600 px-1 rounded">Shift</kbd> to
                        select a range of files
                      </p>
                      <p>• Or drag and drop multiple files directly here</p>
                    </div>
                  </div>

                  {/* Hidden file input with explicit multiple attribute */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*,.png,.jpg,.jpeg"
                    multiple
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </div>

                {useEnhanced && (
                  <div className="mt-4 p-3 bg-blue-900 border border-blue-500 rounded-lg">
                    <p className="text-blue-100 text-sm">
                      <span className="font-semibold">
                        📊 Enhanced Analysis:
                      </span>{" "}
                      Upload 2 screenshots for better analysis.
                      <br />
                      <span className="text-xs">
                        Recommended: Scoreboard (KDA/Gold view) + Match Summary
                        or Damage Report
                      </span>
                    </p>
                  </div>
                )}
              </div>
              <div className="mt-6 text-sm text-gray-500">
                <p>Supported formats: PNG, JPG, JPEG</p>
                <p>Max file size: 10MB per file</p>
              </div>
            </>
          )}
        </div>
      )}

      {/* Analysis Status */}
      {useEnhanced && uploadedScreenshots.length > 0 && (
        <div className="mt-4 p-3 rounded-lg border border-gray-600 bg-gray-700">
          <div className="flex items-center justify-between">
            <span className="text-gray-200">
              {canAnalyze
                ? "✅ Ready for analysis"
                : "⏳ Need more screenshots"}
            </span>
            <span className="text-mlbb-gold font-semibold">
              {uploadedScreenshots.length}/{useEnhanced ? "2" : "1+"}
            </span>
          </div>
          {canAnalyze && (
            <p className="text-xs text-gray-400 mt-1">
              Enhanced analysis with improved accuracy available
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default FileUpload;
