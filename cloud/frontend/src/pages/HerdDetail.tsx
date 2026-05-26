import { useEffect, useState, useRef, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getHerdHorses, getHerds, searchHorses, moveHorse, photoImageUrl, type HorseInHerd, type Herd, type HorseSearchResult } from '../api/client'
import AuthImage from '../components/AuthImage'

export default function HerdDetail() {
  const { id } = useParams<{ id: string }>()
  const [horses, setHorses] = useState<HorseInHerd[]>([])
  const [herd, setHerd] = useState<Herd | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showMoveSearch, setShowMoveSearch] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<HorseSearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [moving, setMoving] = useState(false)
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const searchRef = useRef<HTMLDivElement>(null)

  const herdId = Number(id)

  useEffect(() => {
    if (!id) return
    Promise.all([
      getHerdHorses(herdId),
      getHerds().then(herds => herds.find(h => h.id === herdId) || null),
    ])
      .then(([horses, herd]) => { setHorses(horses); setHerd(herd) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [id])

  // Close search dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setShowMoveSearch(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  const handleSearchInput = useCallback((q: string) => {
    setSearchQuery(q)
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current)
    if (!q.trim()) {
      setSearchResults([])
      return
    }
    searchTimerRef.current = setTimeout(async () => {
      setSearching(true)
      try {
        const r = await searchHorses(q.trim())
        // Filter out horses already in this herd
        setSearchResults(r.filter(h => h.herdName !== herd?.name))
      } catch { /* ignore */ }
      finally { setSearching(false) }
    }, 300)
  }, [herd?.name])

  async function handleMoveHere(horse: HorseSearchResult) {
    if (!confirm(`Move ${horse.name} from ${horse.herdName} to ${herd?.name}?`)) return
    setMoving(true)
    try {
      await moveHorse(horse.id, herdId)
      setShowMoveSearch(false)
      setSearchQuery('')
      setSearchResults([])
      // Refresh the horse list
      const updated = await getHerdHorses(herdId)
      setHorses(updated)
    } catch (e: any) {
      alert(`Failed to move: ${e.message}`)
    } finally {
      setMoving(false)
    }
  }

  if (loading) return <p className="text-gray-500">Loading...</p>
  if (error) return <p className="text-red-600">{error}</p>

  return (
    <div>
      <Link to="/" className="text-sm text-blue-600 hover:underline">&larr; All Herds</Link>
      <div className="flex items-center gap-3 mt-2">
        <h1 className="text-2xl font-semibold text-gray-900">{herd?.name || `Herd #${id}`}</h1>
        <div ref={searchRef} className="relative">
          {!showMoveSearch ? (
            <button
              onClick={() => setShowMoveSearch(true)}
              className="text-xs text-gray-500 hover:text-blue-600 border border-gray-300 rounded px-2 py-1"
            >
              Move horse here...
            </button>
          ) : (
            <div>
              <input
                type="text"
                autoFocus
                placeholder="Search for a horse..."
                value={searchQuery}
                onChange={e => handleSearchInput(e.target.value)}
                className="px-3 py-1.5 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent w-64"
              />
              {searchQuery.trim() && (
                <div className="absolute z-10 mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-lg max-h-64 overflow-y-auto">
                  {searching ? (
                    <p className="px-4 py-3 text-sm text-gray-500">Searching...</p>
                  ) : searchResults.length === 0 ? (
                    <p className="px-4 py-3 text-sm text-gray-500">No horses found</p>
                  ) : (
                    searchResults.map(h => (
                      <button
                        key={h.id}
                        disabled={moving}
                        className="w-full text-left px-4 py-2 text-sm hover:bg-blue-50 disabled:opacity-50"
                        onMouseDown={() => handleMoveHere(h)}
                      >
                        <span className="font-medium text-gray-900">{h.name}</span>
                        <span className="text-gray-500 ml-2">{h.herdName}</span>
                      </button>
                    ))
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      <p className="text-gray-500 text-sm mb-6">{horses.length} horses</p>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {horses.map(horse => (
          <Link
            key={horse.id}
            to={`/horses/${horse.id}`}
            className="block bg-white rounded-lg border border-gray-200 hover:border-blue-300 hover:shadow-sm transition-all overflow-hidden"
          >
            <div className="aspect-square bg-gray-100">
              {horse.thumbnailPhotoId ? (
                <AuthImage
                  src={photoImageUrl(horse.thumbnailPhotoId, 'thumb')}
                  alt=""
                  loading="lazy"
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="flex items-center justify-center h-full text-gray-400 text-sm">No photos</div>
              )}
            </div>
            <div className="p-3">
              <h3 className="font-medium text-gray-900">{horse.name}</h3>
              <p className="text-xs text-gray-500">
                {horse.photoCount} photos, {horse.readyCount} ready
              </p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}
