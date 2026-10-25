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
  const primaryGenre = genres?.split('|').find(Boolean) ?? 'Discover'
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
      .catch((err) => setError(err instanceof ApiError ? err.message : 'Could not load the movie list.'))
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
      setSuccess(`Saved ${value}/5 for “${movie.title}”.`)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not save the rating.')
    } finally {
      setSaving(null)
    }
  }

  return (
    <main className="page wide-page onboarding-page">
      <section className="page-heading onboarding-heading">
        <div className="onboarding-intro">
          <div className="onboarding-meta">
            <p className="eyebrow">BUILD YOUR TASTE PROFILE</p>
            <span className="step-chip">Step 1/2 · Taste profile</span>
          </div>
          <h1>Rate movies for <span>every mood</span></h1>
          <p className="muted">
            Pick 5-10 movies. These ratings build your initial profile
            and you can change them later.
          </p>
        </div>
        <aside className={`progress-card ${isComplete ? 'complete' : ''}`} aria-label="Rating progress">
          <strong>{count} movies rated</strong>
          <span className="progress-message">
            {isComplete
              ? `Minimum of ${TARGET_RATINGS} movies reached`
              : `Rate ${remaining} more`}
          </span>
          <div
            className="progress-track"
            role="progressbar"
            aria-label="Movies rated"
            aria-valuemin={0}
            aria-valuemax={TARGET_RATINGS}
            aria-valuenow={Math.min(count, TARGET_RATINGS)}
          >
            <i style={{ width: `${progress}%` }} />
          </div>
          {isComplete && <Link className="button primary progress-action" to="/room">Continue to the room</Link>}
        </aside>
      </section>
      <div aria-live="assertive">{error && <Alert type="error">{error}</Alert>}</div>
      <div aria-live="polite">{success && <Alert type="success">{success}</Alert>}</div>
      <div className="toolbar onboarding-toolbar">
        <label className="search-field">
          <span className="sr-only">Search movies by title</span>
          <input className="search-input" type="search" placeholder="Search movies..." value={query} onChange={(event) => setQuery(event.target.value)} />
        </label>
        {query && <button className="clear-search" type="button" onClick={() => setQuery('')}>Clear search</button>}
        <span className="movie-count" aria-live="polite">{filtered.length} movies</span>
      </div>
      {loading ? <LoadingState label="Loading movie list..." /> : (
        filtered.length > 0 ? (
          <section className="movie-grid" aria-label="Movies to rate">
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
                    <p>{movie.genres?.replaceAll('|', ' · ') || 'No genres'}</p>
                    <StarRating value={rating} label={`Rating for ${movie.title}`} disabled={isSaving} onChange={(value) => rate(movie, value)} />
                    <div className="rating-status" aria-live="polite">
                      {isSaving
                        ? <span className="saving-label"><i aria-hidden="true" />Saving...</span>
                        : rating
                          ? <span className="saved-label">Saved {rating}/5 stars</span>
                          : <span>Not rated</span>}
                    </div>
                  </div>
                </article>
              )
            })}
          </section>
        ) : (
          <section className="movie-empty-state" role="status">
            <h2>No movies found</h2>
            <p>No results for “{query}”. Try another keyword.</p>
            <button className="button secondary" type="button" onClick={() => setQuery('')}>Clear search</button>
          </section>
        )
      )}
    </main>
  )
}
