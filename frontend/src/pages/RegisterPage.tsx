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
      setError('Mật khẩu xác nhận chưa khớp.')
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
      setError(err instanceof ApiError ? err.message : 'Không thể đăng ký.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="page auth-page">
      <section className="auth-card">
        <div>
          <p className="eyebrow">NEW MEMBER</p>
          <h1>Tạo tài khoản</h1>
          <p className="muted">Mật khẩu tối thiểu 6 ký tự theo contract Backend.</p>
        </div>
        {error && <Alert type="error">{error}</Alert>}
        <form className="form-stack" onSubmit={submit}>
          <label>Email<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" /></label>
          <label>Mật khẩu<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} minLength={6} maxLength={128} required autoComplete="new-password" /></label>
          <label>Xác nhận mật khẩu<input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} minLength={6} required autoComplete="new-password" /></label>
          <button className="button primary full" disabled={loading}>{loading ? 'Đang tạo...' : 'Đăng ký'}</button>
        </form>
        <p className="auth-switch">Đã có tài khoản? <Link to="/login">Đăng nhập</Link></p>
      </section>
    </main>
  )
}
