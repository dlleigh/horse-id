import { useState, useRef, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { searchHorses, type HorseSearchResult } from '../api/client'

export default function HorseSearch() {
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<HorseSearchResult[]>([])
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const ref = useRef<HTMLDivElement>(null)

  const handleSearch = useCallback((q: string) => {
    setQuery(q)
    if (timerRef.current) clearTimeout(timerRef.current)
    if (!q.trim()) {
      setResults([])
      setOpen(false)
      return
    }
    timerRef.current = setTimeout(async () => {
      setLoading(true)
      try {
        const r = await searchHorses(q.trim())
        setResults(r)
        setOpen(true)
      } catch { /* ignore */ }
      finally { setLoading(false) }
    }, 300)
  }, [])

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  return (
    <div ref={ref} className="relative w-64">
      <input
        type="text"
        placeholder="Search horses..."
        value={query}
        onChange={e => handleSearch(e.target.value)}
        onFocus={() => { if (results.length > 0) setOpen(true) }}
        className="w-full px-3 py-1.5 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      />
      {open && (
        <div className="absolute z-10 mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-lg max-h-64 overflow-y-auto">
          {loading ? (
            <p className="px-4 py-3 text-sm text-gray-500">Searching...</p>
          ) : results.length === 0 ? (
            <p className="px-4 py-3 text-sm text-gray-500">No horses found</p>
          ) : (
            results.map(h => (
              <button
                key={h.id}
                className="w-full text-left px-4 py-2 text-sm hover:bg-blue-50 focus:bg-blue-50 focus:outline-none"
                onMouseDown={() => {
                  setOpen(false)
                  setQuery('')
                  setResults([])
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
  )
}
