import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import HerdDetail from './pages/HerdDetail'
import HorseDetail from './pages/HorseDetail'
import Identify from './pages/Identify'

export default function App() {
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
