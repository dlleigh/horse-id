import { useEffect, useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { getHorse, getHerds, moveHorse, patchPhoto, photoImageUrl, type HorseDetail as HorseDetailType, type Herd } from '../api/client'
import AuthImage from '../components/AuthImage'

const STATUS_STYLES: Record<string, string> = {
  pending: 'bg-gray-100 text-gray-600',
  detecting: 'bg-yellow-100 text-yellow-700',
  detected: 'bg-blue-100 text-blue-700',
  extracting: 'bg-yellow-100 text-yellow-700',
  ready: 'bg-green-100 text-green-700',
  error: 'bg-red-100 text-red-700',
}

const DETECTION_STYLES: Record<string, string> = {
  NONE: 'bg-orange-100 text-orange-700',
  MULTIPLE: 'bg-purple-100 text-purple-700',
}

function statusLabel(photo: { processingStatus: string; detectionResult: string | null }): { text: string; style: string } {
  // Photos stuck at "detected" have a non-SINGLE detection result — show that instead
  if (photo.processingStatus === 'detected' && photo.detectionResult && photo.detectionResult !== 'SINGLE') {
    return {
      text: photo.detectionResult.toLowerCase(),
      style: DETECTION_STYLES[photo.detectionResult] || STATUS_STYLES.detected,
    }
  }
  return {
    text: photo.processingStatus,
    style: STATUS_STYLES[photo.processingStatus] || 'bg-gray-100 text-gray-600',
  }
}

export default function HorseDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [horse, setHorse] = useState<HorseDetailType | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [lightboxPhotoId, setLightboxPhotoId] = useState<number | null>(null)
  const [herds, setHerds] = useState<Herd[]>([])
  const [showMoveDropdown, setShowMoveDropdown] = useState(false)
  const [moving, setMoving] = useState(false)

  useEffect(() => {
    if (!id) return
    getHorse(Number(id))
      .then(setHorse)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [id])

  async function toggleExclude(photoId: number, currentlyExcluded: boolean) {
    try {
      await patchPhoto(photoId, !currentlyExcluded)
      setHorse(prev => {
        if (!prev) return prev
        return {
          ...prev,
          photos: prev.photos.map(p =>
            p.id === photoId ? { ...p, excluded: !currentlyExcluded } : p
          ),
        }
      })
    } catch (e: any) {
      alert(`Failed to update: ${e.message}`)
    }
  }

  async function handleMove(destHerdId: number, destHerdName: string) {
    if (!horse) return
    if (!confirm(`Move ${horse.name} to ${destHerdName}?`)) return
    setMoving(true)
    try {
      await moveHorse(horse.id, destHerdId)
      setShowMoveDropdown(false)
      navigate(`/horses/${horse.id}`)
      // Reload horse data to reflect new herd
      const updated = await getHorse(horse.id)
      setHorse(updated)
    } catch (e: any) {
      alert(`Failed to move: ${e.message}`)
    } finally {
      setMoving(false)
    }
  }

  function openMoveDropdown() {
    setShowMoveDropdown(true)
    if (herds.length === 0) {
      getHerds().then(setHerds).catch(() => {})
    }
  }

  if (loading) return <p className="text-gray-500">Loading...</p>
  if (error) return <p className="text-red-600">{error}</p>
  if (!horse) return <p className="text-gray-500">Horse not found</p>

  return (
    <div>
      <Link to={`/herds/${horse.herdId}`} className="text-sm text-blue-600 hover:underline">
        &larr; {horse.herdName}
      </Link>
      <div className="flex items-center gap-3 mt-2">
        <h1 className="text-2xl font-semibold text-gray-900">{horse.name}</h1>
        <div className="relative">
          <button
            onClick={openMoveDropdown}
            className="text-xs text-gray-500 hover:text-blue-600 border border-gray-300 rounded px-2 py-1"
          >
            Move to herd...
          </button>
          {showMoveDropdown && (
            <div className="absolute top-full left-0 mt-1 bg-white border border-gray-200 rounded-md shadow-lg z-10 min-w-48">
              {herds.filter(h => h.id !== horse.herdId).map(h => (
                <button
                  key={h.id}
                  onClick={() => handleMove(h.id, h.name)}
                  disabled={moving}
                  className="block w-full text-left px-3 py-2 text-sm hover:bg-blue-50 disabled:opacity-50"
                >
                  {h.name}
                </button>
              ))}
              {herds.length === 0 && (
                <p className="px-3 py-2 text-sm text-gray-400">Loading...</p>
              )}
              <button
                onClick={() => setShowMoveDropdown(false)}
                className="block w-full text-left px-3 py-2 text-xs text-gray-400 hover:bg-gray-50 border-t"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>
      <p className="text-gray-500 text-sm mb-6">{horse.photos.length} photos</p>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
        {horse.photos.map(photo => (
          <div
            key={photo.id}
            className={`bg-white rounded-lg border overflow-hidden ${
              photo.excluded || photo.detectionResult === 'NONE' || photo.detectionResult === 'MULTIPLE'
                ? 'opacity-40 border-red-200' : 'border-gray-200'
            }`}
          >
            <div
              className="aspect-square bg-gray-100 cursor-pointer"
              onClick={() => setLightboxPhotoId(photo.id)}
            >
              <AuthImage
                src={photoImageUrl(photo.id, 'thumb')}
                alt={photo.filename}
                loading="lazy"
                className="w-full h-full object-cover"
              />
            </div>
            <div className="p-2">
              <div className="flex items-center justify-between gap-1">
                <span className={`text-xs px-1.5 py-0.5 rounded ${statusLabel(photo).style}`}>
                  {statusLabel(photo).text}
                </span>
                <button
                  onClick={() => toggleExclude(photo.id, photo.excluded)}
                  className="text-xs text-gray-500 hover:text-red-600"
                  title={photo.excluded ? 'Include this photo' : 'Exclude this photo'}
                >
                  {photo.excluded ? 'Include' : 'Exclude'}
                </button>
              </div>
              <p className="text-xs text-gray-400 mt-1 truncate" title={photo.filename}>
                {photo.filename}
              </p>
            </div>
          </div>
        ))}
      </div>

      {lightboxPhotoId !== null && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center"
          onClick={() => setLightboxPhotoId(null)}
        >
          <AuthImage
            src={photoImageUrl(lightboxPhotoId)}
            alt=""
            className="max-w-[90vw] max-h-[90vh] object-contain"
            onClick={e => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  )
}
