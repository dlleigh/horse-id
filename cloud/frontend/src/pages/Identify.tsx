import { useEffect, useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import { getHerds, identify, photoImageUrl, type Herd, type Prediction } from '../api/client'
import AuthImage from '../components/AuthImage'

export default function Identify() {
  const [herds, setHerds] = useState<Herd[]>([])
  const [selectedHerdId, setSelectedHerdId] = useState<number | undefined>()
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [predictions, setPredictions] = useState<Prediction[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)

  useEffect(() => {
    getHerds().then(setHerds).catch(() => {})
  }, [])

  function handleFile(f: File | null) {
    setFile(f)
    setPredictions(null)
    setError('')
    if (f) {
      const url = URL.createObjectURL(f)
      setPreview(url)
    } else {
      setPreview(null)
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!file) return
    setLoading(true)
    setError('')
    setPredictions(null)
    try {
      const result = await identify(file, selectedHerdId)
      setPredictions(result.predictions)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f && f.type.startsWith('image/')) handleFile(f)
  }

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-semibold text-gray-900 mb-6">Identify a Horse</h1>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Drop zone */}
        <div
          onDragOver={e => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInput.current?.click()}
          className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-blue-400 transition-colors"
        >
          {preview ? (
            <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded" />
          ) : (
            <div className="text-gray-500">
              <p className="text-lg">Drop a photo here or click to select</p>
              <p className="text-sm mt-1">JPG, PNG, HEIC up to 10MB</p>
            </div>
          )}
          <input
            ref={fileInput}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={e => handleFile(e.target.files?.[0] || null)}
          />
        </div>

        {/* Herd filter */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Filter by herd (optional)
          </label>
          <select
            value={selectedHerdId || ''}
            onChange={e => setSelectedHerdId(e.target.value ? Number(e.target.value) : undefined)}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
          >
            <option value="">All herds</option>
            {herds.map(h => (
              <option key={h.id} value={h.id}>{h.name}</option>
            ))}
          </select>
        </div>

        <button
          type="submit"
          disabled={!file || loading}
          className="w-full py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
        >
          {loading ? 'Identifying...' : 'Identify'}
        </button>
      </form>

      {error && <p className="mt-4 text-red-600">{error}</p>}

      {loading && (
        <p className="mt-4 text-gray-500 text-sm">
          This may take up to a minute on first request (cold start)...
        </p>
      )}

      {predictions && (
        <div className="mt-6">
          <h2 className="text-lg font-medium text-gray-900 mb-3">
            {predictions.length > 0 ? 'Results' : 'No matches found'}
          </h2>
          <div className="space-y-2">
            {predictions.map((p, i) => (
              <Link
                key={i}
                to={`/horses/${p.horse_id}`}
                className="flex items-center gap-4 p-3 bg-white rounded-lg border border-gray-200 hover:border-blue-300 transition-all"
              >
                <div className="w-16 h-16 rounded bg-gray-100 overflow-hidden flex-shrink-0">
                  <AuthImage
                    src={photoImageUrl(p.reference_photo_id)}
                    alt=""
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-900">{p.horse_name}</p>
                  <p className="text-sm text-gray-500">{p.herd_name}</p>
                </div>
                <div className="text-right flex-shrink-0">
                  <p className={`text-lg font-semibold ${p.similarity >= 0.8 ? 'text-green-600' : p.similarity >= 0.5 ? 'text-yellow-600' : 'text-gray-400'}`}>
                    {(p.similarity * 100).toFixed(1)}%
                  </p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
