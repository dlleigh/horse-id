import { useState } from 'react'
import { authClient } from '../lib/auth'
import { triggerReExtract } from '../api/client'

export default function Settings() {
  return (
    <div className="space-y-8 max-w-md">
      <h1 className="text-2xl font-semibold text-gray-900">Settings</h1>
      <ReExtractSection />
      <AddUserForm />
      <ChangePasswordForm />
    </div>
  )
}

function ReExtractSection() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{ deleted: number; reset: number } | null>(null)
  const [error, setError] = useState('')

  async function handleReExtract() {
    if (!confirm('This will delete all embeddings and re-extract them. This may take a while. Continue?')) return
    setError('')
    setResult(null)
    setLoading(true)
    try {
      const r = await triggerReExtract()
      setResult({ deleted: r.deleted, reset: r.reset })
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200 space-y-4">
      <h2 className="text-lg font-medium text-gray-900">Re-extract Embeddings</h2>
      <p className="text-sm text-gray-500">
        Delete all feature embeddings and re-extract them using the current model.
        Use this after deploying a fine-tuned model.
      </p>
      {error && <p className="text-sm text-red-600 bg-red-50 p-2 rounded">{error}</p>}
      {result && (
        <p className="text-sm text-green-600 bg-green-50 p-2 rounded">
          Started: {result.deleted} embeddings deleted, {result.reset} photos queued for extraction.
        </p>
      )}
      <button
        onClick={handleReExtract}
        disabled={loading}
        className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? 'Starting...' : 'Re-extract All'}
      </button>
    </div>
  )
}

function AddUserForm() {
  const [email, setEmail] = useState('')
  const [name, setName] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setSuccess('')
    setLoading(true)
    try {
      const neonAuthUrl = import.meta.env.VITE_NEON_AUTH_URL as string
      const res = await fetch(`${neonAuthUrl}/sign-up/email`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, name }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => null)
        setError(body?.message ?? `Failed to create user (${res.status})`)
      } else {
        setSuccess(`User ${email} created`)
        setEmail('')
        setName('')
        setPassword('')
      }
    } catch (err: any) {
      setError(err.message || 'Failed to create user')
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="bg-white p-6 rounded-lg border border-gray-200 space-y-4">
      <h2 className="text-lg font-medium text-gray-900">Add User</h2>
      {error && <p className="text-sm text-red-600 bg-red-50 p-2 rounded">{error}</p>}
      {success && <p className="text-sm text-green-600 bg-green-50 p-2 rounded">{success}</p>}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
        <input
          type="email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          required
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
        <input
          type="text"
          value={name}
          onChange={e => setName(e.target.value)}
          required
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
        <input
          type="password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
          minLength={8}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>
      <button
        type="submit"
        disabled={loading}
        className="w-full py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? 'Creating...' : 'Create User'}
      </button>
    </form>
  )
}

function ChangePasswordForm() {
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setSuccess('')

    if (newPassword !== confirmPassword) {
      setError('Passwords do not match')
      return
    }

    setLoading(true)
    try {
      const result = await authClient.changePassword({ currentPassword, newPassword })
      if (result.error) {
        setError(result.error.message ?? 'Failed to change password')
      } else {
        setSuccess('Password changed')
        setCurrentPassword('')
        setNewPassword('')
        setConfirmPassword('')
      }
    } catch (err: any) {
      setError(err.message || 'Failed to change password')
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="bg-white p-6 rounded-lg border border-gray-200 space-y-4">
      <h2 className="text-lg font-medium text-gray-900">Change Password</h2>
      {error && <p className="text-sm text-red-600 bg-red-50 p-2 rounded">{error}</p>}
      {success && <p className="text-sm text-green-600 bg-green-50 p-2 rounded">{success}</p>}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Current Password</label>
        <input
          type="password"
          value={currentPassword}
          onChange={e => setCurrentPassword(e.target.value)}
          required
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">New Password</label>
        <input
          type="password"
          value={newPassword}
          onChange={e => setNewPassword(e.target.value)}
          required
          minLength={8}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Confirm New Password</label>
        <input
          type="password"
          value={confirmPassword}
          onChange={e => setConfirmPassword(e.target.value)}
          required
          minLength={8}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>
      <button
        type="submit"
        disabled={loading}
        className="w-full py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? 'Changing...' : 'Change Password'}
      </button>
    </form>
  )
}
