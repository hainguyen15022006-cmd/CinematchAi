import { FormEvent, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Alert } from '../components/Alert'
import { api, ApiError, setToken } from '../services/api'

export function LoginPage() {
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function submit(event: FormEvent) {
    event.preventDefault()
    setError('')
    setLoading(true)
    try {
      const token = await api.login(email, password)
      setToken(token.access_token)
      navigate('/onboarding')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not sign in.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="page auth-page">
      <section className="auth-card">
        <div>
          <p className="eyebrow">WELCOME BACK</p>
          <h1>Sign in</h1>
          <p className="muted">Keep rating movies and join your group's pick.</p>
        </div>
        {error && <Alert type="error">{error}</Alert>}
        <form className="form-stack" onSubmit={submit}>
          <label>Email<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" /></label>
          <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} minLength={6} required autoComplete="current-password" /></label>
          <button className="button primary full" disabled={loading}>{loading ? 'Signing in...' : 'Sign in'}</button>
        </form>
        <p className="auth-switch">Don't have an account? <Link to="/register">Register</Link></p>
      </section>
    </main>
  )
}
