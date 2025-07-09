import { useState } from 'react'

const ResultsDisplay = ({ result, onReset, sessionInfo, debugInfo, warnings }) => {
  const [showDebugInfo, setShowDebugInfo] = useState(false)
  const getSeverityColor = (type) => {
    switch (type) {
      case 'critical':
        return 'bg-red-900 border-red-500 text-red-100'
      case 'warning':
        return 'bg-yellow-900 border-yellow-500 text-yellow-100'
      case 'info':
        return 'bg-blue-900 border-blue-500 text-blue-100'
      default:
        return 'bg-gray-900 border-gray-500 text-gray-100'
    }
  }

  const getSeverityIcon = (type) => {
    switch (type) {
      case 'critical':
        return '🔴'
      case 'warning':
        return '⚠️'
      case 'info':
        return '💡'
      default:
        return '📝'
    }
  }

  const getRatingColor = (rating) => {
    switch (rating.toLowerCase()) {
      case 'excellent':
        return 'text-green-400'
      case 'good':
        return 'text-blue-400'
      case 'average':
        return 'text-yellow-400'
      case 'poor':
        return 'text-red-400'
      default:
        return 'text-gray-400'
    }
  }

  const criticalFeedback = result.feedback.filter(item => item.type === 'critical')
  const warningFeedback = result.feedback.filter(item => item.type === 'warning')
  const infoFeedback = result.feedback.filter(item => item.type === 'info')

  return (
    <div className="bg-gray-800 rounded-lg p-8 shadow-xl">
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-2xl font-bold text-mlbb-gold">Analysis Results</h2>
        <button
          onClick={onReset}
          className="bg-mlbb-blue hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
        >
          Analyze New Screenshot
        </button>
      </div>

      {/* Overall Rating */}
      <div className="mb-8 p-6 bg-gray-700 rounded-lg">
        <h3 className="text-lg font-semibold text-white mb-2">Overall Performance</h3>
        <p className={`text-2xl font-bold ${getRatingColor(result.overall_rating)}`}>
          {result.overall_rating}
        </p>
      </div>

      {/* Mental Boost */}
      <div className="mb-8 p-6 bg-gradient-to-r from-mlbb-purple to-purple-600 rounded-lg">
        <h3 className="text-lg font-semibold text-white mb-2">Mental Boost</h3>
        <p className="text-white">{result.mental_boost}</p>
      </div>

      {/* Feedback Sections */}
      <div className="space-y-6">
        {criticalFeedback.length > 0 && (
          <div>
            <h3 className="text-lg font-semibold text-red-400 mb-3">
              Critical Issues ({criticalFeedback.length})
            </h3>
            <div className="space-y-3">
              {criticalFeedback.map((item, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-lg border-l-4 ${getSeverityColor(item.type)}`}
                >
                  <div className="flex items-start space-x-3">
                    <span className="text-lg">{getSeverityIcon(item.type)}</span>
                    <div>
                      <p className="font-semibold mb-1">{item.category}</p>
                      <p>{item.message}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {warningFeedback.length > 0 && (
          <div>
            <h3 className="text-lg font-semibold text-yellow-400 mb-3">
              Warnings ({warningFeedback.length})
            </h3>
            <div className="space-y-3">
              {warningFeedback.map((item, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-lg border-l-4 ${getSeverityColor(item.type)}`}
                >
                  <div className="flex items-start space-x-3">
                    <span className="text-lg">{getSeverityIcon(item.type)}</span>
                    <div>
                      <p className="font-semibold mb-1">{item.category}</p>
                      <p>{item.message}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {infoFeedback.length > 0 && (
          <div>
            <h3 className="text-lg font-semibold text-blue-400 mb-3">
              Tips & Insights ({infoFeedback.length})
            </h3>
            <div className="space-y-3">
              {infoFeedback.map((item, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-lg border-l-4 ${getSeverityColor(item.type)}`}
                >
                  <div className="flex items-start space-x-3">
                    <span className="text-lg">{getSeverityIcon(item.type)}</span>
                    <div>
                      <p className="font-semibold mb-1">{item.category}</p>
                      <p>{item.message}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {result.feedback.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-400 text-lg">No feedback items found.</p>
        </div>
      )}

      {/* Session Information */}
      {sessionInfo && (
        <div className="mt-8 p-6 bg-gray-700 rounded-lg">
          <h3 className="text-lg font-semibold text-white mb-4">Session Information</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Session ID:</span>
              <span className="ml-2 text-white font-mono">{sessionInfo.session_id?.substring(0, 8)}...</span>
            </div>
            <div>
              <span className="text-gray-400">Screenshot Type:</span>
              <span className="ml-2 text-white capitalize">{sessionInfo.screenshot_type}</span>
            </div>
            <div>
              <span className="text-gray-400">Type Confidence:</span>
              <span className="ml-2 text-white">{(sessionInfo.type_confidence * 100).toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-gray-400">Screenshots:</span>
              <span className="ml-2 text-white">{sessionInfo.screenshot_count}</span>
            </div>
            {sessionInfo.session_complete && (
              <div className="col-span-2">
                <span className="text-green-400">✓ Session Complete</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Warnings */}
      {warnings && warnings.length > 0 && (
        <div className="mt-6 p-4 bg-yellow-900 border-l-4 border-yellow-500 rounded-lg">
          <h4 className="text-yellow-300 font-semibold mb-2">Analysis Warnings</h4>
          <ul className="text-yellow-100 text-sm space-y-1">
            {warnings.map((warning, index) => (
              <li key={index} className="flex items-start">
                <span className="mr-2">⚠️</span>
                <span>{warning}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Debug Information */}
      {debugInfo && Object.keys(debugInfo).length > 0 && (
        <div className="mt-6">
          <button
            onClick={() => setShowDebugInfo(!showDebugInfo)}
            className="text-gray-400 hover:text-white text-sm flex items-center"
          >
            <span className="mr-2">{showDebugInfo ? '▼' : '▶'}</span>
            Debug Information
          </button>
          
          {showDebugInfo && (
            <div className="mt-3 p-4 bg-gray-900 rounded-lg text-sm">
              {debugInfo.detected_keywords && (
                <div className="mb-3">
                  <span className="text-gray-400">Detected Keywords:</span>
                  <span className="ml-2 text-white">{debugInfo.detected_keywords.join(', ')}</span>
                </div>
              )}
              
              {debugInfo.hero_suggestions && debugInfo.hero_suggestions.length > 0 && (
                <div className="mb-3">
                  <span className="text-gray-400">Hero Suggestions:</span>
                  <ul className="ml-4 mt-1">
                    {debugInfo.hero_suggestions.slice(0, 3).map((suggestion, index) => (
                      <li key={index} className="text-white">
                        {suggestion[0]} ({(suggestion[1] * 100).toFixed(1)}%)
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              
              {debugInfo.hero_debug && (
                <div className="mb-3">
                  <span className="text-gray-400">Hero Detection:</span>
                  <ul className="ml-4 mt-1 text-sm">
                    <li className="text-white">
                      Strategies: {debugInfo.hero_debug.strategies_tried?.join(', ') || 'None'}
                    </li>
                    {debugInfo.hero_debug.manual_override && (
                      <li className="text-green-400">Manual override used</li>
                    )}
                    {debugInfo.hero_debug.error && (
                      <li className="text-red-400">Error: {debugInfo.hero_debug.error}</li>
                    )}
                  </ul>
                </div>
              )}
              
              <div className="text-xs text-gray-500 mt-3">
                <pre className="whitespace-pre-wrap">{JSON.stringify(debugInfo, null, 2)}</pre>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default ResultsDisplay