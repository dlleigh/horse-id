import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getHorse, patchPhoto, photoImageUrl, type HorseDetail as HorseDetailType } from '../api/client'

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
  const [horse, setHorse] = useState<HorseDetailType | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [lightboxPhotoId, setLightboxPhotoId] = useState<number | null>(null)

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

  if (loading) return <p className="text-gray-500">Loading...</p>
  if (error) return <p className="text-red-600">{error}</p>
  if (!horse) return <p className="text-gray-500">Horse not found</p>

  return (
    <div>
      <Link to={`/herds/${horse.herdId}`} className="text-sm text-blue-600 hover:underline">
        &larr; {horse.herdName}
      </Link>
      <h1 className="text-2xl font-semibold text-gray-900 mt-2">{horse.name}</h1>
      <p className="text-gray-500 text-sm mb-6">{horse.photos.length} photos</p>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
        {horse.photos.map(photo => (
          <div
            key={photo.id}
            className={`bg-white rounded-lg border overflow-hidden ${
              photo.excluded ? 'opacity-40 border-red-200' : 'border-gray-200'
            }`}
          >
            <div
              className="aspect-square bg-gray-100 cursor-pointer"
              onClick={() => setLightboxPhotoId(photo.id)}
            >
              <img
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
          <img
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
