import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Alert } from '../components/Alert'
import { LoadingState } from '../components/LoadingState'
import { StarRating } from '../components/StarRating'
import { api, ApiError } from '../services/api'
import '../styles/onboarding.css'
import type { Movie } from '../types'

const TARGET_RATINGS = 5

const GENRE_THEMES: Record<string, string> = {
  Action: 'action',
  Adventure: 'adventure',
  Animation: 'animation',
  Children: 'children',
  Comedy: 'comedy',
  Crime: 'crime',
  Drama: 'drama',
  Horror: 'horror',
  Romance: 'romance',
  'Sci-Fi': 'sci-fi',
  Thriller: 'thriller',
}

function getMovieVisual(genres?: string | null) {
  const primaryGenre = genres?.split('|').find(Boolean) ?? 'Khám phá'
  return {
    primaryGenre,
    theme: GENRE_THEMES[primaryGenre] ?? 'default',
  }
}

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
  const remaining = Math.max(0, TARGET_RATINGS - count)
  const isComplete = count >= TARGET_RATINGS

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
    <main className="page wide-page onboarding-page">
      <section className="page-heading onboarding-heading">
        <div className="onboarding-intro">
          <div className="onboarding-meta">
            <p className="eyebrow">HIỂU GU CỦA BẠN</p>
            <span className="step-chip">Bước 1/2 · Hồ sơ sở thích</span>
          </div>
          <h1>Chấm phim cho <span>mọi khoảnh khắc</span></h1>
          <p className="muted">
            Chọn từ 5–10 phim. Các lựa chọn này tạo hồ sơ ban đầu
            và bạn vẫn có thể thay đổi sau.
          </p>
        </div>
        <aside className={`progress-card ${isComplete ? 'complete' : ''}`} aria-label="Tiến độ chấm phim">
          <strong>Đã chấm {count} phim</strong>
          <span className="progress-message">
            {isComplete
              ? `Đã đạt yêu cầu tối thiểu ${TARGET_RATINGS} phim`
              : `Cần chấm thêm ${remaining} phim`}
          </span>
          <div
            className="progress-track"
            role="progressbar"
            aria-label="Số phim đã chấm"
            aria-valuemin={0}
            aria-valuemax={TARGET_RATINGS}
            aria-valuenow={Math.min(count, TARGET_RATINGS)}
          >
            <i style={{ width: `${progress}%` }} />
          </div>
          {isComplete && <Link className="button primary progress-action" to="/room">Tiếp tục vào phòng</Link>}
        </aside>
      </section>
      <div aria-live="assertive">{error && <Alert type="error">{error}</Alert>}</div>
      <div aria-live="polite">{success && <Alert type="success">{success}</Alert>}</div>
      <div className="toolbar onboarding-toolbar">
        <label className="search-field">
          <span className="sr-only">Tìm phim theo tên</span>
          <input className="search-input" type="search" placeholder="Tìm theo tên phim..." value={query} onChange={(event) => setQuery(event.target.value)} />
        </label>
        {query && <button className="clear-search" type="button" onClick={() => setQuery('')}>Xóa tìm kiếm</button>}
        <span className="movie-count" aria-live="polite">{filtered.length} phim</span>
      </div>
      {loading ? <LoadingState label="Đang lấy danh sách phim..." /> : (
        filtered.length > 0 ? (
          <section className="movie-grid" aria-label="Danh sách phim để chấm">
            {filtered.map((movie) => {
              const rating = ratings[movie.movielens_id]
              const isSaving = saving === movie.movielens_id
              const visual = getMovieVisual(movie.genres)
              return (
                <article className={`movie-card ${rating ? 'rated' : ''} ${isSaving ? 'saving' : ''}`} key={movie.movielens_id}>
                  <div className="poster-placeholder" data-genre-theme={visual.theme} aria-hidden="true">
                    <span className="genre-kicker">{visual.primaryGenre}</span>
                    <span className="movie-monogram">{movie.title.trim().charAt(0).toUpperCase()}</span>
                    <span className="release-year">{movie.release_year ?? '—'}</span>
                  </div>
                  <div className="movie-content">
                    <span className="movie-id">MovieLens #{movie.movielens_id}</span>
                    <h2>{movie.title}</h2>
                    <p>{movie.genres?.replaceAll('|', ' · ') || 'Chưa có thể loại'}</p>
                    <StarRating value={rating} label={`Đánh giá phim ${movie.title}`} disabled={isSaving} onChange={(value) => rate(movie, value)} />
                    <div className="rating-status" aria-live="polite">
                      {isSaving
                        ? <span className="saving-label"><i aria-hidden="true" />Đang lưu...</span>
                        : rating
                          ? <span className="saved-label">Đã lưu {rating}/5 sao</span>
                          : <span>Chưa chấm</span>}
                    </div>
                  </div>
                </article>
              )
            })}
          </section>
        ) : (
          <section className="movie-empty-state" role="status">
            <h2>Không tìm thấy phim phù hợp</h2>
            <p>Không có kết quả cho “{query}”. Hãy thử một từ khóa khác.</p>
            <button className="button secondary" type="button" onClick={() => setQuery('')}>Xóa tìm kiếm</button>
          </section>
        )
      )}
    </main>
  )
}
