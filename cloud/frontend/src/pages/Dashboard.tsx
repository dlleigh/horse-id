import { useEffect, useRef, useState, useCallback } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { getHerds, getStats, triggerSync, triggerProcessing, getErrorPhotos, retryPhoto, searchHorses, type Herd, type Stats, type ErrorPhoto, type HorseSearchResult } from '../api/client'

export default function Dashboard() {
  const [herds, setHerds] = useState<Herd[]>([])
  const [stats, setStats] = useState<Stats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [errorPhotos, setErrorPhotos] = useState<ErrorPhoto[]>([])
  const [showErrors, setShowErrors] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const navigate = useNavigate()

  // Horse search
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<HorseSearchResult[]>([])
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchLoading, setSearchLoading] = useState(false)
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const searchRef = useRef<HTMLDivElement>(null)

  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query)
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current)
    if (!query.trim()) {
      setSearchResults([])
      setSearchOpen(false)
      return
    }
    searchTimerRef.current = setTimeout(async () => {
      setSearchLoading(true)
      try {
        const results = await searchHorses(query.trim())
        setSearchResults(results)
        setSearchOpen(true)
      } catch { /* ignore */ }
      finally { setSearchLoading(false) }
    }, 300)
  }, [])

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setSearchOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

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

  async function handleSync(mode?: 'full' | 'incremental') {
    setError('')
    try {
      await triggerSync(mode)
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
            onClick={() => handleSync()}
            disabled={isBusy}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {stats?.syncStatus === 'running' ? 'Syncing...' : isProcessing ? 'Processing...' : 'Sync'}
          </button>
          <button
            onClick={() => handleSync('full')}
            disabled={isBusy}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
          >
            Full Sync
          </button>
        </div>
      </div>

      {stats && stats.total > 0 && (
        <div className="mb-6 p-3 bg-gray-50 border border-gray-200 rounded-lg">
          <div className="flex items-center gap-4 text-sm">
            {stats.syncStatus === 'running' && (
              <span className="flex items-center gap-1.5 text-blue-700 font-medium">
                <span className="inline-block w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                Syncing{stats.syncProgressTotal && stats.syncProgressDone !== null && stats.syncProgressDone < stats.syncProgressTotal
                  ? ` (${stats.syncProgressDone}/${stats.syncProgressTotal} herds listed${stats.syncFilesScanned ? `, ${stats.syncFilesScanned.toLocaleString()} files found` : ''})`
                  : stats.syncFilesScanned ? ` (${stats.syncFilesScanned.toLocaleString()} files scanned)` : ''}
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
            {stats.activeWorkers > 0 && (
              <span className="text-indigo-600">
                <span className="font-medium">{stats.activeWorkers}</span> {stats.activeWorkers === 1 ? 'worker' : 'workers'}
              </span>
            )}
            {stats.error > 0 && (
              <button
                onClick={async () => {
                  if (!showErrors) {
                    const photos = await getErrorPhotos()
                    setErrorPhotos(photos)
                  }
                  setShowErrors(!showErrors)
                }}
                className="text-red-600 hover:text-red-800 underline decoration-dotted"
              >
                <span className="font-medium">{stats.error}</span> errors
              </button>
            )}
          </div>
          {stats.lastSync && (
            <>
              <p className="text-xs text-gray-400 mt-1.5">
                Last sync: {stats.lastSync.filesScanned.toLocaleString()} scanned
                {stats.lastSync.filesAdded > 0 && `, ${stats.lastSync.filesAdded.toLocaleString()} added`}
                {stats.lastSync.filesRemoved > 0 && `, ${stats.lastSync.filesRemoved.toLocaleString()} removed`}
                {stats.lastSync.filesMoved > 0 && `, ${stats.lastSync.filesMoved.toLocaleString()} moved`}
              </p>
              {stats.lastSync.warnings?.length > 0 && (
                <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded text-xs text-yellow-800">
                  <p className="font-medium mb-1">Sync warnings ({stats.lastSync.warnings.length}):</p>
                  <ul className="list-disc list-inside space-y-0.5">
                    {stats.lastSync.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {showErrors && errorPhotos.length > 0 && (
        <div className="mb-6 p-3 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-red-800">Error Photos</h3>
            <button onClick={() => setShowErrors(false)} className="text-red-400 hover:text-red-600 text-xs">
              Hide
            </button>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-red-700 text-xs">
                <th className="pb-1 pr-4">Filename</th>
                <th className="pb-1 pr-4">Horse</th>
                <th className="pb-1 pr-4">Herd</th>
                <th className="pb-1"></th>
              </tr>
            </thead>
            <tbody>
              {errorPhotos.map(p => (
                <tr key={p.id} className="text-red-900">
                  <td className="py-0.5 pr-4 font-mono text-xs">
                    <a href={`https://drive.google.com/file/d/${p.drive_file_id}/view`} target="_blank" rel="noopener noreferrer" className="text-red-700 underline hover:text-red-900">{p.filename}</a>
                  </td>
                  <td className="py-0.5 pr-4">{p.horse_name}</td>
                  <td className="py-0.5 pr-4">{p.herd_name}</td>
                  <td className="py-0.5">
                    <button
                      onClick={async () => {
                        await retryPhoto(p.id)
                        setErrorPhotos(prev => prev.filter(ep => ep.id !== p.id))
                        setStats(prev => prev ? { ...prev, error: prev.error - 1, pending: prev.pending + 1 } : prev)
                      }}
                      className="text-red-500 hover:text-red-800 text-xs"
                      title="Retry"
                    >
                      ↻
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Horse search */}
      <div ref={searchRef} className="relative mb-4">
        <input
          type="text"
          placeholder="Search horses..."
          value={searchQuery}
          onChange={e => handleSearch(e.target.value)}
          onFocus={() => { if (searchResults.length > 0) setSearchOpen(true) }}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        {searchOpen && (
          <div className="absolute z-10 mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-lg max-h-64 overflow-y-auto">
            {searchLoading ? (
              <p className="px-4 py-3 text-sm text-gray-500">Searching...</p>
            ) : searchResults.length === 0 ? (
              <p className="px-4 py-3 text-sm text-gray-500">No horses found</p>
            ) : (
              searchResults.map(h => (
                <button
                  key={h.id}
                  className="w-full text-left px-4 py-2 text-sm hover:bg-blue-50 focus:bg-blue-50 focus:outline-none"
                  onMouseDown={() => {
                    setSearchOpen(false)
                    setSearchQuery('')
                    setSearchResults([])
                    navigate(`/horses/${h.id}`)
                  }}
                >
                  <span className="font-medium text-gray-900">{h.name}</span>
                  <span className="text-gray-500 ml-2">{h.herdName}</span>
                </button>
              ))
            )}
          </div>
        )}
      </div>

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
