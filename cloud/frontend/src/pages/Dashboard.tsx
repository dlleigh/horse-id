import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { getHerds, getStats, triggerSync, triggerProcessing, type Herd, type Stats } from '../api/client'

export default function Dashboard() {
  const [herds, setHerds] = useState<Herd[]>([])
  const [stats, setStats] = useState<Stats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Fetch initial data
  useEffect(() => {
    Promise.all([getHerds(), getStats()])
      .then(([h, s]) => { setHerds(h); setStats(s) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  // Poll while there's activity (sync running, or photos pending/processing)
  useEffect(() => {
    const shouldPoll = stats && (
      stats.syncStatus === 'running' ||
      stats.pending > 0 ||
      stats.detecting > 0 ||
      stats.detected > 0 ||
      stats.extracting > 0
    )

    if (shouldPoll && !pollRef.current) {
      pollRef.current = setInterval(async () => {
        try {
          const [h, s] = await Promise.all([getHerds(), getStats()])
          setHerds(h)
          setStats(s)
        } catch { /* ignore poll errors */ }
      }, 2000)
    } else if (!shouldPoll && pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [stats])

  async function handleSync() {
    setError('')
    try {
      await triggerSync()
      const s = await getStats()
      setStats(s)
    } catch (e: any) {
      setError(e.message)
    }
  }

  async function handleProcess() {
    setError('')
    try {
      await triggerProcessing()
      const s = await getStats()
      setStats(s)
    } catch (e: any) {
      setError(e.message)
    }
  }

  if (loading) return <p className="text-gray-500">Loading...</p>
  if (error) return <p className="text-red-600">{error}</p>

  const totalHorses = herds.reduce((sum, h) => sum + Number(h.horseCount), 0)
  const isProcessing = stats ? (stats.pending + stats.detecting + stats.detected + stats.extracting) > 0 : false
  const isBusy = stats?.syncStatus === 'running' || isProcessing

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900">Dashboard</h1>
          <p className="text-gray-500 text-sm">
            {herds.length} herds, {totalHorses} horses
          </p>
        </div>
        <div className="flex gap-2">
          {isProcessing && stats?.syncStatus !== 'running' && (
            <button
              onClick={handleProcess}
              className="px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700"
            >
              Reprocess
            </button>
          )}
          <button
            onClick={handleSync}
            disabled={isBusy}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {stats?.syncStatus === 'running' ? 'Syncing...' : isProcessing ? 'Processing...' : 'Sync from Drive'}
          </button>
        </div>
      </div>

      {stats && stats.total > 0 && (
        <div className="mb-6 p-3 bg-gray-50 border border-gray-200 rounded-lg">
          <div className="flex items-center gap-4 text-sm">
            {stats.syncStatus === 'running' && (
              <span className="flex items-center gap-1.5 text-blue-700 font-medium">
                <span className="inline-block w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                Syncing{stats.syncProgressTotal ? ` (${stats.syncProgressDone}/${stats.syncProgressTotal} horses)` : ''}
              </span>
            )}
            {(stats.pending > 0 || stats.detecting > 0) && (
              <span className="flex items-center gap-1.5 text-yellow-700 font-medium">
                <span className="inline-block w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
                Detecting ({stats.pending + stats.detecting} remaining)
              </span>
            )}
            {(stats.detected > 0 || stats.extracting > 0) && stats.detecting === 0 && stats.pending === 0 && (
              <span className="flex items-center gap-1.5 text-purple-700 font-medium">
                <span className="inline-block w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
                Extracting ({stats.detected + stats.extracting} remaining)
              </span>
            )}
            <span className="text-gray-600">
              <span className="font-medium">{stats.total}</span> photos
            </span>
            <span className="text-green-700">
              <span className="font-medium">{stats.ready}</span> ready
            </span>
            {stats.error > 0 && (
              <span className="text-red-600">
                <span className="font-medium">{stats.error}</span> errors
              </span>
            )}
          </div>
          {stats.lastSync && (
            <p className="text-xs text-gray-400 mt-1.5">
              Last sync: {stats.lastSync.filesScanned.toLocaleString()} scanned
              {stats.lastSync.filesAdded > 0 && `, ${stats.lastSync.filesAdded.toLocaleString()} added`}
              {stats.lastSync.filesRemoved > 0 && `, ${stats.lastSync.filesRemoved.toLocaleString()} removed`}
              {stats.lastSync.filesMoved > 0 && `, ${stats.lastSync.filesMoved.toLocaleString()} moved`}
            </p>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {herds.map(herd => (
          <Link
            key={herd.id}
            to={`/herds/${herd.id}`}
            className="block p-5 bg-white rounded-lg border border-gray-200 hover:border-blue-300 hover:shadow-sm transition-all"
          >
            <h2 className="text-lg font-medium text-gray-900">{herd.name}</h2>
            <p className="text-sm text-gray-500 mt-1">
              {herd.horseCount} horses, {herd.photoCount} photos
            </p>
          </Link>
        ))}
      </div>
    </div>
  )
}
