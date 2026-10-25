import { FormEvent, useCallback, useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { Alert } from '../components/Alert'
import { LoadingState } from '../components/LoadingState'
import { api, ApiError } from '../services/api'
import '../styles/run.css'
import type {
  AggregationStrategy,
  Room,
  UserProfile,
} from '../types'

const STRATEGY_OPTIONS: { value: AggregationStrategy; label: string; hint: string }[] = [
  {
    value: 'average',
    label: 'Average',
    hint: 'Every member counts the same.',
  },
  {
    value: 'least_misery',
    label: 'Least Misery',
    hint: 'Protects whoever likes the movie least.',
  },
  {
    value: 'average_without_misery',
    label: 'Average Without Misery',
    hint: 'Average, but drops movies someone would dislike.',
  },
]

const LOBBY_POLL_MS = 4000

export function RoomPage() {
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const code = (params.get('code') ?? '').toUpperCase()

  const [roomName, setRoomName] = useState('')
  const [joinCode, setJoinCode] = useState('')
  const [room, setRoom] = useState<Room | null>(null)
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [strategy, setStrategy] = useState<AggregationStrategy>('average')
  const [topK, setTopK] = useState(10)
  const [loading, setLoading] = useState(Boolean(code))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const isHost = Boolean(profile && room && profile.id === room.host_id)
  const everyoneReady =
    Boolean(room) &&
    room!.members.length >= 2 &&
    room!.members.every((member) => member.is_ready)

  const openRoom = useCallback((next: Room) => {
    setRoom(next)
    setParams({ code: next.code }, { replace: true })
  }, [setParams])

  const refresh = useCallback(async () => {
    if (!code) return
    setRoom(await api.getRoom(code))
  }, [code])

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const me = await api.me()
        if (!cancelled) setProfile(me)
        if (code) {
          const loaded = await api.getRoom(code)
          if (!cancelled) setRoom(loaded)
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof ApiError ? err.message : 'Could not load the room.',
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
  }, [code])

  // Keep the lobby fresh so members see each other get ready.
  useEffect(() => {
    if (!code || !room) return
    const timer = window.setInterval(() => {
      void refresh().catch(() => undefined)
    }, LOBBY_POLL_MS)
    return () => window.clearInterval(timer)
  }, [code, room, refresh])

  async function act<T>(action: () => Promise<T>, fallback: string) {
    setError('')
    setBusy(true)
    try {
      return await action()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : fallback)
      return null
    } finally {
      setBusy(false)
    }
  }

  async function create(event: FormEvent) {
    event.preventDefault()
    const created = await act(
      () => api.createRoom(roomName),
      'Could not create the room.',
    )
    if (created) openRoom(created)
  }

  async function join(event: FormEvent) {
    event.preventDefault()
    const joined = await act(
      () => api.joinRoom(joinCode),
      'Could not join the room.',
    )
    if (joined) openRoom(joined)
  }

  async function toggleReady() {
    if (!room) return
    await act(async () => {
      await api.toggleReady(room.id)
      await refresh()
    }, 'Could not update your ready status.')
  }

  async function startRun() {
    if (!room) return
    const run = await act(async () => {
      await api.updateRoomConstraints(room.id, {
        strategy,
        top_k: topK,
      })
      return api.triggerRecommendation(room.id)
    }, 'Could not start the recommendation run.')
    if (run) navigate(`/rooms/${room.code}/runs/${run.id}`)
  }

  function openExistingRun() {
    if (room?.latest_run_id) {
      navigate(`/rooms/${room.code}/runs/${room.latest_run_id}`)
    }
  }

  if (loading) return <LoadingState label="Loading the room..." />

  return (
    <main className="page wide-page run-page">
      <section className="page-heading">
        <div>
          <p className="eyebrow">STEP 2/2 · GROUP ROOM</p>
          <h1>{room ? room.name || 'CineMatch room' : 'Create or join a room'}</h1>
          <p className="muted">
            Two to five people. Everyone rates a few movies first, then the
            host starts the group Top-K.
          </p>
        </div>
      </section>

      {error && <Alert type="error">{error}</Alert>}

      {!room && (
        <section className="room-actions">
          <form className="panel" onSubmit={create}>
            <h2>Create room</h2>
            <label>
              Room name
              <input
                value={roomName}
                onChange={(event) => setRoomName(event.target.value)}
                placeholder="Movie night"
              />
            </label>
            <button className="button primary" disabled={busy}>
              Create room
            </button>
          </form>
          <form className="panel" onSubmit={join}>
            <h2>Join room</h2>
            <label>
              Room code
              <input
                value={joinCode}
                onChange={(event) =>
                  setJoinCode(event.target.value.toUpperCase())
                }
                placeholder="ABC123"
                required
              />
            </label>
            <button className="button secondary" disabled={busy}>
              Join
            </button>
          </form>
        </section>
      )}

      {room && (
        <>
          <section className="room-lobby">
            <div className="room-code">
              <span>ROOM CODE</span>
              <strong>{room.code}</strong>
            </div>
            <div>
              <h2>Share this code with your group</h2>
              <p>
                Status: <strong>{room.status}</strong> ·{' '}
                {room.members.length}{' '}
                {room.members.length === 1 ? 'member' : 'members'} ·{' '}
                {everyoneReady ? 'everyone is ready' : 'waiting for members'}
              </p>
            </div>
            <button
              className="button primary"
              onClick={toggleReady}
              disabled={busy}
            >
              Toggle Ready
            </button>
            <div className="member-list">
              {room.members.map((member) => (
                <div key={member.id}>
                  <span>
                    {member.user?.email || `User #${member.user_id}`}
                    {member.user_id === room.host_id ? ' · host' : ''}
                  </span>
                  <strong className={member.is_ready ? 'ready' : 'not-ready'}>
                    {member.is_ready ? 'READY' : 'WAITING'}
                  </strong>
                </div>
              ))}
            </div>
          </section>

          {room.latest_run_id && (
            <section className="panel">
              <h2>This room already has a Top-K</h2>
              <p className="muted">
                Voting may still be open. Open it to vote or to see the result.
              </p>
              <button className="button primary" onClick={openExistingRun}>
                Open the vote
              </button>
            </section>
          )}

          {isHost ? (
            <section className="panel">
              <h2>Start the group Top-K</h2>
              <p className="muted">
                The model scores each member separately, then these settings
                decide how the scores become one group ranking.
              </p>
              <label>
                Aggregation strategy
                <select
                  value={strategy}
                  onChange={(event) =>
                    setStrategy(event.target.value as AggregationStrategy)
                  }
                >
                  {STRATEGY_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <p className="muted">
                {
                  STRATEGY_OPTIONS.find(
                    (option) => option.value === strategy,
                  )?.hint
                }
              </p>
              <label>
                How many movies
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={topK}
                  onChange={(event) =>
                    setTopK(
                      Math.min(20, Math.max(1, Number(event.target.value))),
                    )
                  }
                />
              </label>
              <button
                className="button primary"
                onClick={startRun}
                disabled={busy || !everyoneReady}
              >
                Generate group Top-K
              </button>
              {!everyoneReady && (
                <p className="muted">
                  Everyone in the room has to be ready first.
                </p>
              )}
            </section>
          ) : (
            <section className="panel">
              <h2>Waiting for the host</h2>
              <p className="muted">
                Mark yourself ready. The host starts the group Top-K, and this
                page opens the vote as soon as it exists.
              </p>
            </section>
          )}
        </>
      )}
    </main>
  )
}
