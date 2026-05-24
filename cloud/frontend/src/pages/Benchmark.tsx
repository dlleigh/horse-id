import { useState, useEffect } from 'react'
import { getHerds, runBenchmark, type Herd, type BenchmarkResult } from '../api/client'

function pct(n: number): string {
  return (n * 100).toFixed(1) + '%'
}

function accuracyColor(n: number): string {
  if (n >= 0.8) return 'text-green-600'
  if (n >= 0.5) return 'text-yellow-600'
  return 'text-red-600'
}

export default function Benchmark() {
  const [herds, setHerds] = useState<Herd[]>([])
  const [selectedHerdId, setSelectedHerdId] = useState<string>('')
  const [testPct, setTestPct] = useState(20)
  const [result, setResult] = useState<BenchmarkResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    getHerds().then(setHerds).catch(() => {})
  }, [])

  async function handleRun() {
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const r = await runBenchmark(
        selectedHerdId ? Number(selectedHerdId) : undefined,
        testPct / 100,
      )
      setResult(r)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Benchmark failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Matching Performance Test</h1>

      {/* Controls */}
      <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
        <div className="flex flex-wrap items-end gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Herd
            </label>
            <select
              value={selectedHerdId}
              onChange={e => setSelectedHerdId(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-2 text-sm"
              disabled={loading}
            >
              <option value="">All herds</option>
              {herds.map(h => (
                <option key={h.id} value={h.id}>{h.name}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Test percentage
            </label>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min={5}
                max={50}
                step={5}
                value={testPct}
                onChange={e => setTestPct(Number(e.target.value))}
                className="w-32"
                disabled={loading}
              />
              <span className="text-sm text-gray-600 w-10">{testPct}%</span>
            </div>
          </div>

          <button
            onClick={handleRun}
            disabled={loading}
            className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Running…' : 'Run Benchmark'}
          </button>
        </div>

        {loading && (
          <p className="mt-4 text-sm text-gray-500">
            Running benchmark — matching each test photo against the training set…
          </p>
        )}

        {error && (
          <p className="mt-4 text-sm text-red-600">{error}</p>
        )}
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Summary card */}
          <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Results</h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-gray-500">Rank-1 Accuracy</p>
                <p className={`text-2xl font-bold ${accuracyColor(result.rank1Accuracy)}`}>
                  {pct(result.rank1Accuracy)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Top-5 Accuracy</p>
                <p className={`text-2xl font-bold ${accuracyColor(result.top5Accuracy)}`}>
                  {pct(result.top5Accuracy)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Avg Top Match Score</p>
                <p className="text-2xl font-bold text-gray-900">
                  {result.avgTopMatchSimilarity.toFixed(3)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Avg Correct Match Score</p>
                <p className="text-2xl font-bold text-gray-900">
                  {result.avgCorrectMatchSimilarity.toFixed(3)}
                </p>
              </div>
            </div>
            <div className="mt-4 flex flex-wrap gap-x-6 gap-y-1 text-sm text-gray-500">
              <span>Test photos: {result.testCount}</span>
              <span>Training photos: {result.trainingCount}</span>
              <span>Horses evaluated: {result.horsesEvaluated} / {result.horsesTotal}</span>
              <span>Duration: {(result.durationMs / 1000).toFixed(1)}s</span>
              <span>Seed: {result.seed}</span>
            </div>
          </div>

          {/* Per-horse breakdown */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Per-Horse Breakdown</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200 text-left">
                    <th className="pb-2 font-medium text-gray-500">Horse</th>
                    <th className="pb-2 font-medium text-gray-500">Herd</th>
                    <th className="pb-2 font-medium text-gray-500 text-right">Training Photos</th>
                    <th className="pb-2 font-medium text-gray-500 text-right">Test Photos</th>
                    <th className="pb-2 font-medium text-gray-500 text-right">Rank-1 Correct</th>
                    <th className="pb-2 font-medium text-gray-500 text-right">Avg Similarity</th>
                    <th className="pb-2 font-medium text-gray-500">Confused With</th>
                  </tr>
                </thead>
                <tbody>
                  {result.perHorseResults.map(h => {
                    const horseAcc = h.testPhotos > 0 ? h.rank1Correct / h.testPhotos : 0
                    return (
                      <tr key={h.horseId} className="border-b border-gray-100">
                        <td className="py-2 font-medium text-gray-900">{h.horseName}</td>
                        <td className="py-2 text-gray-600">{h.herdName}</td>
                        <td className="py-2 text-right text-gray-600">{h.trainingPhotos}</td>
                        <td className="py-2 text-right text-gray-600">{h.testPhotos}</td>
                        <td className={`py-2 text-right font-medium ${accuracyColor(horseAcc)}`}>
                          {h.rank1Correct}/{h.testPhotos}
                        </td>
                        <td className="py-2 text-right text-gray-600">
                          {h.avgSimilarity > 0 ? h.avgSimilarity.toFixed(3) : '—'}
                        </td>
                        <td className="py-2 text-gray-600">
                          {h.confusedWith.length > 0
                            ? h.confusedWith.map(c =>
                                `${c.horseName} (${c.herdName})${c.count > 1 ? ` ×${c.count}` : ''}`
                              ).join(', ')
                            : '—'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
