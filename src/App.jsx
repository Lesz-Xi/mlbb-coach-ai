import { useState } from 'react'
import FileUpload from './components/FileUpload'
import ResultsDisplay from './components/ResultsDisplay'

function App() {
  const [analysisResult, setAnalysisResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleFileUpload = async (file) => {
    setIsLoading(true)
    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Analysis failed')
      }

      const result = await response.json()
      setAnalysisResult(result)
    } catch (error) {
      console.error('Error analyzing screenshot:', error)
      // Handle error appropriately
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    setAnalysisResult(null)
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
        </header>

        <main className="max-w-4xl mx-auto">
          {!analysisResult ? (
            <FileUpload onFileUpload={handleFileUpload} isLoading={isLoading} />
          ) : (
            <ResultsDisplay 
              result={analysisResult} 
              onReset={handleReset}
            />
          )}
        </main>
      </div>
    </div>
  )
}

export default App