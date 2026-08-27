import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { AppHeader } from './components/AppHeader'
import { ProtectedRoute } from './components/ProtectedRoute'
import { HomePage } from './pages/HomePage'
import { LoginPage } from './pages/LoginPage'
import { NotFoundPage } from './pages/NotFoundPage'
import { OnboardingPage } from './pages/OnboardingPage'
import { RecommendationsPage } from './pages/RecommendationsPage'
import { RegisterPage } from './pages/RegisterPage'
import { RoomPage } from './pages/RoomPage'

export default function App() {
  return (
    <BrowserRouter>
      <AppHeader />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/recommendations" element={<RecommendationsPage />} />
        <Route path="/onboarding" element={<ProtectedRoute><OnboardingPage /></ProtectedRoute>} />
        <Route path="/room" element={<ProtectedRoute><RoomPage /></ProtectedRoute>} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </BrowserRouter>
  )
}
