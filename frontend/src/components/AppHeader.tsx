import { NavLink, useNavigate } from 'react-router-dom'
import { clearToken, getToken } from '../services/api'

export function AppHeader() {
  const navigate = useNavigate()
  const loggedIn = Boolean(getToken())

  function logout() {
    clearToken()
    navigate('/login')
  }

  return (
    <header className="app-header">
      <NavLink to="/" className="brand" aria-label="CineMatch home">
        <span className="brand-mark">C</span>
        <span>CineMatch</span>
      </NavLink>
      <nav className="main-nav" aria-label="Điều hướng chính">
        {loggedIn && <NavLink to="/onboarding">Chấm phim</NavLink>}
        {loggedIn && <NavLink to="/room">Phòng nhóm</NavLink>}
        <NavLink to="/recommendations">Top 10</NavLink>
        {loggedIn ? (
          <button className="text-button" onClick={logout}>Đăng xuất</button>
        ) : (
          <NavLink to="/login">Đăng nhập</NavLink>
        )}
      </nav>
    </header>
  )
}
