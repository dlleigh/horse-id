import { Link, Outlet } from 'react-router-dom'
import { authClient } from '../lib/auth'

export default function Layout() {
  async function handleSignOut() {
    await authClient.signOut()
    window.location.reload()
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200 px-4 py-3">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <Link to="/" className="text-xl font-semibold text-gray-900">
            Horse ID
          </Link>
          <div className="flex items-center gap-4">
            <Link to="/" className="text-gray-600 hover:text-gray-900">Herds</Link>
            <Link to="/identify" className="text-gray-600 hover:text-gray-900">Identify</Link>
            <Link to="/settings" className="text-gray-600 hover:text-gray-900">Settings</Link>
            <button
              onClick={handleSignOut}
              className="text-gray-400 hover:text-gray-600 text-sm"
            >
              Sign out
            </button>
          </div>
        </div>
      </nav>
      <main className="max-w-6xl mx-auto px-4 py-6">
        <Outlet />
      </main>
    </div>
  )
}
