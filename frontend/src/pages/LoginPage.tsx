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
      setError(err instanceof ApiError ? err.message : 'Không thể đăng nhập.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="page auth-page">
      <section className="auth-card">
        <div>
          <p className="eyebrow">WELCOME BACK</p>
          <h1>Đăng nhập</h1>
          <p className="muted">Tiếp tục chấm phim và tham gia lựa chọn của nhóm.</p>
        </div>
        {error && <Alert type="error">{error}</Alert>}
        <form className="form-stack" onSubmit={submit}>
          <label>Email<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" /></label>
          <label>Mật khẩu<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} minLength={6} required autoComplete="current-password" /></label>
          <button className="button primary full" disabled={loading}>{loading ? 'Đang đăng nhập...' : 'Đăng nhập'}</button>
        </form>
        <p className="auth-switch">Chưa có tài khoản? <Link to="/register">Đăng ký</Link></p>
      </section>
    </main>
  )
}
