import { useEffect, useMemo, useState } from 'react'
import { Alert } from '../components/Alert'
import { LoadingState } from '../components/LoadingState'
import { StarRating } from '../components/StarRating'
import { api, ApiError } from '../services/api'
import type { Movie } from '../types'

const TARGET_RATINGS = 5

export function OnboardingPage() {
  const [movies, setMovies] = useState<Movie[]>([])
  const [ratings, setRatings] = useState<Record<number, number>>({})
  const [saving, setSaving] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [query, setQuery] = useState('')

  useEffect(() => {
    api.movies(50, 0)
      .then(setMovies)
      .catch((err) => setError(err instanceof ApiError ? err.message : 'Không tải được phim.'))
      .finally(() => setLoading(false))
  }, [])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return q ? movies.filter((movie) => movie.title.toLowerCase().includes(q)) : movies
  }, [movies, query])

  const count = Object.keys(ratings).length
  const progress = Math.min(100, Math.round((count / TARGET_RATINGS) * 100))

  async function rate(movie: Movie, value: number) {
    setError('')
    setSuccess('')
    setSaving(movie.movielens_id)
    try {
      await api.rateMovie(movie.movielens_id, value)
      setRatings((current) => ({ ...current, [movie.movielens_id]: value }))
      setSuccess(`Đã lưu ${value}/5 cho “${movie.title}”.`)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Không lưu được rating.')
    } finally {
      setSaving(null)
    }
  }

  return (
    <main className="page wide-page">
      <section className="page-heading">
        <div><p className="eyebrow">ONBOARDING</p><h1>Chấm những phim bạn đã xem</h1><p className="muted">Rating được gửi trực tiếp tới <code>POST /ratings</code> bằng MovieLens ID.</p></div>
        <div className="progress-card"><strong>{count}/{TARGET_RATINGS}</strong><span>phim đã chấm trong phiên này</span><div className="progress-track"><i style={{ width: `${progress}%` }} /></div></div>
      </section>
      {error && <Alert type="error">{error}</Alert>}
      {success && <Alert type="success">{success}</Alert>}
      <div className="toolbar"><input className="search-input" placeholder="Tìm phim..." value={query} onChange={(e) => setQuery(e.target.value)} /><span>{filtered.length} phim</span></div>
      {loading ? <LoadingState label="Đang lấy danh sách phim..." /> : (
        <section className="movie-grid">
          {filtered.map((movie) => (
            <article className="movie-card" key={movie.movielens_id}>
              <div className="poster-placeholder"><span>{movie.release_year ?? '—'}</span></div>
              <div className="movie-content">
                <span className="movie-id">MovieLens #{movie.movielens_id}</span>
                <h2>{movie.title}</h2>
                <p>{movie.genres?.replaceAll('|', ' · ') || 'Chưa có thể loại'}</p>
                <StarRating value={ratings[movie.movielens_id]} disabled={saving === movie.movielens_id} onChange={(value) => rate(movie, value)} />
              </div>
            </article>
          ))}
        </section>
      )}
    </main>
  )
}
