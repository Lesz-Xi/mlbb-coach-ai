import { useState } from 'react'
import FileUpload from './components/FileUpload'
import ResultsDisplay from './components/ResultsDisplay'

function App() {
  const [analysisResult, setAnalysisResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [useEnhanced, setUseEnhanced] = useState(true)
  const [heroOverride, setHeroOverride] = useState('')
  const [playerIGN, setPlayerIGN] = useState('Lesz XVII')

  const handleFileUpload = async (file) => {
    setIsLoading(true)
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('ign', playerIGN)
      
      // Add session ID if we have one
      if (sessionId) {
        formData.append('session_id', sessionId)
      }
      
      // Add hero override if specified
      if (heroOverride.trim()) {
        formData.append('hero_override', heroOverride.trim())
      }

      const endpoint = useEnhanced ? '/api/analyze-enhanced' : '/api/analyze'
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Analysis failed')
      }

      const result = await response.json()
      setAnalysisResult(result)
      
      // Store session ID for multi-screenshot processing
      if (result.session_info?.session_id) {
        setSessionId(result.session_info.session_id)
      }
    } catch (error) {
      console.error('Error analyzing screenshot:', error)
      // Handle error appropriately
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    setAnalysisResult(null)
    setSessionId(null)
    setHeroOverride('')
  }

  const handleNewSession = () => {
    setSessionId(null)
    setAnalysisResult(null)
    setHeroOverride('')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-mlbb-dark to-gray-900">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-mlbb-gold mb-4">
            MLBB Coach AI
          </h1>
          <p className="text-gray-300 text-lg">
            AI-powered coaching analysis for Mobile Legends: Bang Bang
          </p>
          
          {/* Enhanced Analysis Toggle */}
          <div className="mt-6 flex justify-center items-center space-x-4">
            <label className="flex items-center space-x-2 text-gray-300">
              <input
                type="checkbox"
                checked={useEnhanced}
                onChange={(e) => setUseEnhanced(e.target.checked)}
                className="rounded"
              />
              <span>Enhanced Analysis (Multi-Screenshot)</span>
            </label>
          </div>
          
          {/* Player Settings */}
          <div className="mt-6 max-w-2xl mx-auto space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Player IGN
                </label>
                <input
                  type="text"
                  value={playerIGN}
                  onChange={(e) => setPlayerIGN(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-mlbb-gold focus:border-transparent"
                  placeholder="Enter your in-game name"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Hero Override (Optional)
                </label>
                <input
                  type="text"
                  value={heroOverride}
                  onChange={(e) => setHeroOverride(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-mlbb-gold focus:border-transparent"
                  placeholder="e.g., Fredrinn, Kagura..."
                />
              </div>
            </div>
            <p className="text-xs text-gray-400 text-center">
              Use hero override if automatic detection fails
            </p>
          </div>
          
          {/* Session Info */}
          {sessionId && (
            <div className="mt-4 p-3 bg-gray-800 rounded-lg inline-block">
              <p className="text-sm text-gray-300">
                Session: <span className="text-mlbb-gold font-mono">{sessionId.substring(0, 8)}...</span>
                {analysisResult?.session_info && (
                  <span className="ml-2">
                    ({analysisResult.session_info.screenshot_count} screenshots)
                  </span>
                )}
              </p>
              <button
                onClick={handleNewSession}
                className="mt-2 text-xs bg-gray-700 text-gray-300 px-3 py-1 rounded hover:bg-gray-600"
              >
                New Session
              </button>
            </div>
          )}
        </header>

        <main className="max-w-4xl mx-auto">
          {!analysisResult ? (
            <FileUpload onFileUpload={handleFileUpload} isLoading={isLoading} />
          ) : (
            <ResultsDisplay 
              result={analysisResult} 
              onReset={handleReset}
              sessionInfo={analysisResult.session_info}
              debugInfo={analysisResult.debug_info}
              warnings={analysisResult.warnings}
            />
          )}
        </main>
        <div className="fixed bottom-8 right-8">
          <button
            onClick={() => document.querySelector('input[type="file"]').click()}
            disabled={isLoading}
            className="bg-mlbb-gold text-mlbb-dark font-semibold py-3 px-6 rounded-full shadow-lg hover:bg-yellow-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Analyzing...' : sessionId ? 'Add Another Screenshot' : 'Analyze Screenshot'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default App