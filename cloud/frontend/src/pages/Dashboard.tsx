import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getHerds, triggerSync, getSyncStatus, type Herd, type SyncRun } from '../api/client'

export default function Dashboard() {
  const [herds, setHerds] = useState<Herd[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [sync, setSync] = useState<SyncRun | null>(null)
  const [syncing, setSyncing] = useState(false)

  useEffect(() => {
    getHerds()
      .then(setHerds)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  async function handleSync() {
    setSyncing(true)
    setError('')
    try {
      const { syncRunId } = await triggerSync()
      const poll = setInterval(async () => {
        const status = await getSyncStatus(syncRunId)
        setSync(status)
        if (status.status !== 'running') {
          clearInterval(poll)
          setSyncing(false)
          // Refresh herds after sync
          getHerds().then(setHerds)
        }
      }, 2000)
    } catch (e: any) {
      setError(e.message)
      setSyncing(false)
    }
  }

  if (loading) return <p className="text-gray-500">Loading...</p>
  if (error) return <p className="text-red-600">{error}</p>

  const totalHorses = herds.reduce((sum, h) => sum + Number(h.horseCount), 0)
  const totalPhotos = herds.reduce((sum, h) => sum + Number(h.photoCount), 0)

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900">Dashboard</h1>
          <p className="text-gray-500 text-sm">
            {herds.length} herds, {totalHorses} horses, {totalPhotos} photos
          </p>
        </div>
        <button
          onClick={handleSync}
          disabled={syncing}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {syncing ? 'Syncing...' : 'Sync from Drive'}
        </button>
      </div>

      {sync && (
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg text-sm">
          <p className="font-medium text-blue-900">
            Sync {sync.status === 'running' ? 'in progress' : sync.status}
          </p>
          <p className="text-blue-700">
            Scanned: {sync.filesScanned} | Added: {sync.filesAdded} | Removed: {sync.filesRemoved} | Moved: {sync.filesMoved}
          </p>
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
