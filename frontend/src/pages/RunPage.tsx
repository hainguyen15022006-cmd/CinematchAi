import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Alert } from '../components/Alert'
import { LoadingState } from '../components/LoadingState'
import { api, ApiError } from '../services/api'
import '../styles/run.css'
import type {
  AggregationStrategy,
  RecommendationResponse,
  Room,
  RunResult,
  UserProfile,
} from '../types'

const STRATEGY_LABELS: Record<AggregationStrategy, string> = {
  average: 'Average',
  least_misery: 'Least Misery',
  average_without_misery: 'Average Without Misery',
}

const RESULT_POLL_MS = 4000

export function RunPage() {
  const { code = '', runId = '' } = useParams()
  const numericRunId = Number(runId)

  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [room, setRoom] = useState<Room | null>(null)
  const [recommendations, setRecommendations] =
    useState<RecommendationResponse | null>(null)
  const [result, setResult] = useState<RunResult | null>(null)
  const [myVote, setMyVote] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  const isHost = Boolean(profile && room && profile.id === room.host_id)

  const loadResult = useCallback(async () => {
    try {
      setResult(await api.runResult(numericRunId))
      return true
    } catch (err) {
      // 409 simply means voting is still open.
      if (err instanceof ApiError && err.status === 409) return false
      throw err
    }
  }, [numericRunId])

  useEffect(() => {
    if (!Number.isInteger(numericRunId) || numericRunId <= 0) {
      setError('Invalid run address.')
      setLoading(false)
      return
    }

    let cancelled = false
    async function load() {
      try {
        const [me, loadedRoom, loadedRecommendations] = await Promise.all([
          api.me(),
          api.getRoom(code),
          api.runRecommendations(numericRunId),
        ])
        if (cancelled) return
        setProfile(me)
        setRoom(loadedRoom)
        setRecommendations(loadedRecommendations)
        await loadResult()
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof ApiError
              ? err.message
              : 'Could not load this recommendation run.',
          )
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [code, numericRunId, loadResult])

  // While voting is open, watch for the host closing it.
  useEffect(() => {
    if (loading || result || error) return
    const timer = window.setInterval(() => {
      void loadResult().catch(() => undefined)
    }, RESULT_POLL_MS)
    return () => window.clearInterval(timer)
  }, [loading, result, error, loadResult])

  async function vote(movieId: number) {
    setError('')
    setNotice('')
    setBusy(true)
    try {
      await api.castVote(numericRunId, movieId, 1)
      setMyVote(movieId)
      setNotice('Your vote is saved. You can change it until voting closes.')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not vote.')
    } finally {
      setBusy(false)
    }
  }

  async function finalize() {
    setError('')
    setNotice('')
    setBusy(true)
    try {
      setResult(await api.finalizeRun(numericRunId))
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : 'Could not close voting.',
      )
    } finally {
      setBusy(false)
    }
  }

  const voteTally = useMemo(() => {
    const tally = new Map<number, number>()
    for (const castVote of result?.votes ?? []) {
      tally.set(
        castVote.movie_id,
        (tally.get(castVote.movie_id) ?? 0) + castVote.vote_value,
      )
    }
    return tally
  }, [result])

  if (loading) return <LoadingState label="Loading the group Top-K..." />

  if (error && !recommendations) {
    return (
      <main className="page wide-page run-page">
        <Alert type="error">{error}</Alert>
        <Link className="button secondary" to="/room">
          Back to the room
        </Link>
      </main>
    )
  }

  const strategy = recommendations?.strategy ?? 'average'

  return (
    <main className="page wide-page run-page">
      <section className="page-heading run-heading">
        <div>
          <p className="eyebrow">STEP 2/2 · GROUP DECISION</p>
          <h1>{result ? 'The group picked a movie' : 'Vote for tonight'}</h1>
          <p className="muted">
            Room <strong>{code}</strong> · aggregated with{' '}
            <strong>{STRATEGY_LABELS[strategy]}</strong> ·{' '}
            {recommendations?.recommendations.length ?? 0} candidates
          </p>
        </div>
        {isHost && !result && (
          <button
            className="button primary"
            onClick={finalize}
            disabled={busy}
          >
            Close voting and pick the winner
          </button>
        )}
        {!isHost && !result && (
          <p className="waiting-note" role="status">
            The host closes the vote when everyone has chosen.
          </p>
        )}
      </section>

      {error && <Alert type="error">{error}</Alert>}
      {notice && <Alert type="success">{notice}</Alert>}

      {result && (
        <section className="result-card">
          <div className="winner-block">
            <span className="winner-badge">WINNER</span>
            <h2>{result.winner_movie?.title ?? 'No winner recorded'}</h2>
            <p className="muted">
              {result.winner_movie?.genres?.replaceAll('|', ' · ') ||
                'No genres'}
              {result.winner_movie?.release_year
                ? ` · ${result.winner_movie.release_year}`
                : ''}
            </p>
          </div>
          <dl className="result-numbers">
            <div>
              <dt>Group score</dt>
              <dd>{result.group_score?.toFixed(2) ?? '—'}</dd>
            </div>
            <div>
              <dt>Disagreement</dt>
              <dd>{result.disagreement?.toFixed(2) ?? '—'}</dd>
            </div>
            <div>
              <dt>Votes cast</dt>
              <dd>{result.votes.length}</dd>
            </div>
          </dl>
          <Link className="button secondary" to="/room">
            Start another room
          </Link>
        </section>
      )}

      <section className="vote-grid">
        {recommendations?.recommendations.map((movie) => {
          const tally = voteTally.get(movie.movie_id) ?? 0
          const isWinner =
            result?.winner_movie?.movielens_id === movie.movie_id
          const isMine = myVote === movie.movie_id
          return (
            <article
              className={`vote-card${isMine ? ' mine' : ''}${
                isWinner ? ' winner' : ''
              }`}
              key={movie.movie_id}
            >
              <header>
                <span className="rank-badge">#{movie.rank}</span>
                <div>
                  <h2>{movie.title}</h2>
                  <p>{movie.genres.join(' · ') || 'No genres'}</p>
                </div>
                {movie.misery_warning && (
                  <span className="warning-badge">
                    Someone would not enjoy this
                  </span>
                )}
              </header>

              <div className="score-row">
                <div>
                  <span>Group</span>
                  <strong>{movie.group_score.toFixed(2)}</strong>
                </div>
                <div>
                  <span>Lowest member</span>
                  <strong>{movie.minimum_score.toFixed(2)}</strong>
                </div>
                <div>
                  <span>Disagreement</span>
                  <strong>{movie.disagreement.toFixed(2)}</strong>
                </div>
              </div>

              <ul className="member-predictions">
                {movie.member_scores.map((member) => (
                  <li key={member.user_id}>
                    <span>
                      {member.display_name || `User ${member.user_id}`}
                    </span>
                    <strong>{member.predicted_score.toFixed(1)}</strong>
                  </li>
                ))}
              </ul>

              <footer>
                {result ? (
                  <span className="vote-count">
                    {tally} {tally === 1 ? 'vote' : 'votes'}
                  </span>
                ) : (
                  <button
                    className={`button ${isMine ? 'primary' : 'secondary'}`}
                    onClick={() => vote(movie.movie_id)}
                    disabled={busy}
                  >
                    {isMine ? 'Your vote' : 'Vote for this'}
                  </button>
                )}
              </footer>
            </article>
          )
        })}
      </section>
    </main>
  )
}
