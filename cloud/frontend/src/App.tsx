import { Routes, Route } from 'react-router-dom'
import { authClient } from './lib/auth'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import HerdDetail from './pages/HerdDetail'
import HorseDetail from './pages/HorseDetail'
import Identify from './pages/Identify'
import Login from './pages/Login'

export default function App() {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const session = (authClient as any).useSession()

  if (session.isPending) {
    return <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <p className="text-gray-500">Loading...</p>
    </div>
  }

  if (!session.data) {
    return <Login onLogin={() => session.refetch()} />
  }

  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/herds/:id" element={<HerdDetail />} />
        <Route path="/horses/:id" element={<HorseDetail />} />
        <Route path="/identify" element={<Identify />} />
      </Route>
    </Routes>
  )
}
