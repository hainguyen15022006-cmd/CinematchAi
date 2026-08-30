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
    let active = true

    Promise.all([api.movies(50, 0), api.myRatings()])
      .then(([loadedMovies, savedRatings]) => {
        if (!active) return
        const savedByMovie = savedRatings.reduce<Record<number, number>>(
          (current, item) => {
            current[item.movie_id] = item.rating
            return current
          },
          {},
        )
        setMovies(loadedMovies)
        setRatings(savedByMovie)
      })
      .catch((err) => {
        if (active) {
          setError(err instanceof ApiError ? err.message : 'Could not load movies.')
        }
      })
      .finally(() => {
        if (active) setLoading(false)
      })

    return () => {
      active = false
    }
  }, [])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return q ? movies.filter((movie) => movie.title.toLowerCase().includes(q)) : movies
  }, [movies, query])

  const count = Object.keys(ratings).length
  const progress = Math.min(100, Math.round((count / TARGET_RATINGS) * 100))
  const hasReachedTarget = count >= TARGET_RATINGS
  const remaining = Math.max(0, TARGET_RATINGS - count)

  async function rate(movie: Movie, value: number) {
    setError('')
    setSuccess('')
    setSaving(movie.movielens_id)
    try {
      await api.rateMovie(movie.movielens_id, value)
      setRatings((current) => ({ ...current, [movie.movielens_id]: value }))
      setSuccess(`Saved ${value}/5 for “${movie.title}”.`)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not save rating.')
    } finally {
      setSaving(null)
    }
  }

  return (
    <main className="page wide-page">
      <section className="page-heading">
        <div><p className="eyebrow">ONBOARDING</p><h1>Rate the movies you have watched</h1><p className="muted">Ratings are sent directly to <code>POST /ratings</code> using the MovieLens ID.</p></div>
        <div className="progress-card">
          <strong>{hasReachedTarget ? `${count} movies rated` : `${count}/${TARGET_RATINGS} movies`}</strong>
          <span>{hasReachedTarget ? `Minimum of ${TARGET_RATINGS} movies reached` : `Rate ${remaining} more`}</span>
          <div className="progress-track"><i style={{ width: `${progress}%` }} /></div>
        </div>
      </section>
      {error && <Alert type="error">{error}</Alert>}
      {success && <Alert type="success">{success}</Alert>}
      <div className="toolbar"><input className="search-input" placeholder="Search movies..." value={query} onChange={(e) => setQuery(e.target.value)} /><span>{filtered.length} movies</span></div>
      {loading ? <LoadingState label="Loading movie list..." /> : (
        <section className="movie-grid">
          {filtered.map((movie) => (
            <article className="movie-card" key={movie.movielens_id}>
              <div className="poster-placeholder"><span>{movie.release_year ?? '—'}</span></div>
              <div className="movie-content">
                <span className="movie-id">MovieLens #{movie.movielens_id}</span>
                <h2>{movie.title}</h2>
                <p>{movie.genres?.replaceAll('|', ' · ') || 'No genres'}</p>
                <StarRating value={ratings[movie.movielens_id]} disabled={saving === movie.movielens_id} onChange={(value) => rate(movie, value)} />
              </div>
            </article>
          ))}
        </section>
      )}
    </main>
  )
}
