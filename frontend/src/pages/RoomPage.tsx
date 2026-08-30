import { FormEvent, useState } from 'react'
import { Alert } from '../components/Alert'
import { api, ApiError } from '../services/api'
import type { Room } from '../types'

export function RoomPage() {
  const [roomName, setRoomName] = useState('')
  const [joinCode, setJoinCode] = useState('')
  const [room, setRoom] = useState<Room | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function execute(action: () => Promise<Room>) {
    setError('')
    setLoading(true)
    try { setRoom(await action()) }
    catch (err) { setError(err instanceof ApiError ? err.message : 'Room action failed.') }
    finally { setLoading(false) }
  }

  function create(event: FormEvent) { event.preventDefault(); void execute(() => api.createRoom(roomName)) }
  function join(event: FormEvent) { event.preventDefault(); void execute(() => api.joinRoom(joinCode)) }
  async function ready() {
    if (!room) return
    setError('')
    setLoading(true)
    try {
      await api.toggleReady(room.id)
      setRoom(await api.getRoom(room.code))
    } catch (err) { setError(err instanceof ApiError ? err.message : 'Could not update ready status.') }
    finally { setLoading(false) }
  }

  return (
    <main className="page wide-page">
      <section className="page-heading"><div><p className="eyebrow">GROUP ROOM</p><h1>Create or join a room</h1><p className="muted">Supporting flow using the Backend's real room API.</p></div></section>
      {error && <Alert type="error">{error}</Alert>}
      <section className="room-actions">
        <form className="panel" onSubmit={create}><h2>Create room</h2><label>Room name<input value={roomName} onChange={(e) => setRoomName(e.target.value)} placeholder="Movie night" /></label><button className="button primary" disabled={loading}>Create room</button></form>
        <form className="panel" onSubmit={join}><h2>Join room</h2><label>Room code<input value={joinCode} onChange={(e) => setJoinCode(e.target.value.toUpperCase())} placeholder="ABC123" required /></label><button className="button secondary" disabled={loading}>Join</button></form>
      </section>
      {room && <section className="room-lobby"><div className="room-code"><span>ROOM CODE</span><strong>{room.code}</strong></div><div><h2>{room.name || 'CineMatch room'}</h2><p>Status: <strong>{room.status}</strong> · {room.members.length} {room.members.length === 1 ? 'member' : 'members'}</p></div><button className="button primary" onClick={ready} disabled={loading}>Toggle Ready</button><div className="member-list">{room.members.map((member) => <div key={member.id}><span>{member.user?.email || `User #${member.user_id}`}</span><strong className={member.is_ready ? 'ready' : 'not-ready'}>{member.is_ready ? 'READY' : 'WAITING'}</strong></div>)}</div></section>}
    </main>
  )
}
