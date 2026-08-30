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
      <nav className="main-nav" aria-label="Main navigation">
        {loggedIn && <NavLink to="/onboarding">Rate movies</NavLink>}
        {loggedIn && <NavLink to="/room">Group room</NavLink>}
        <NavLink to="/recommendations">Top 10</NavLink>
        {loggedIn ? (
          <button className="text-button" onClick={logout}>Sign out</button>
        ) : (
          <NavLink to="/login">Sign in</NavLink>
        )}
      </nav>
    </header>
  )
}
