import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getHerdHorses, getHerds, photoImageUrl, type HorseInHerd, type Herd } from '../api/client'

export default function HerdDetail() {
  const { id } = useParams<{ id: string }>()
  const [horses, setHorses] = useState<HorseInHerd[]>([])
  const [herd, setHerd] = useState<Herd | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!id) return
    const herdId = Number(id)
    Promise.all([
      getHerdHorses(herdId),
      getHerds().then(herds => herds.find(h => h.id === herdId) || null),
    ])
      .then(([horses, herd]) => { setHorses(horses); setHerd(herd) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [id])

  if (loading) return <p className="text-gray-500">Loading...</p>
  if (error) return <p className="text-red-600">{error}</p>

  return (
    <div>
      <Link to="/" className="text-sm text-blue-600 hover:underline">&larr; All Herds</Link>
      <h1 className="text-2xl font-semibold text-gray-900 mt-2">{herd?.name || `Herd #${id}`}</h1>
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
                <img
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
