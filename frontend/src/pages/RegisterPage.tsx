import { FormEvent, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Alert } from '../components/Alert'
import { api, ApiError, setToken } from '../services/api'

export function RegisterPage() {
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (password !== confirm) {
      setError('Passwords do not match.')
      return
    }
    setError('')
    setLoading(true)
    try {
      await api.register(email, password)
      const token = await api.login(email, password)
      setToken(token.access_token)
      navigate('/onboarding')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not register.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="page auth-page">
      <section className="auth-card">
        <div>
          <p className="eyebrow">NEW MEMBER</p>
          <h1>Create an account</h1>
          <p className="muted">Password must be at least 6 characters, per the Backend contract.</p>
        </div>
        {error && <Alert type="error">{error}</Alert>}
        <form className="form-stack" onSubmit={submit}>
          <label>Email<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" /></label>
          <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} minLength={6} maxLength={128} required autoComplete="new-password" /></label>
          <label>Confirm password<input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} minLength={6} required autoComplete="new-password" /></label>
          <button className="button primary full" disabled={loading}>{loading ? 'Creating...' : 'Register'}</button>
        </form>
        <p className="auth-switch">Already have an account? <Link to="/login">Sign in</Link></p>
      </section>
    </main>
  )
}
