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
    catch (err) { setError(err instanceof ApiError ? err.message : 'Không thao tác được với phòng.') }
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
    } catch (err) { setError(err instanceof ApiError ? err.message : 'Không đổi được trạng thái ready.') }
    finally { setLoading(false) }
  }

  return (
    <main className="page wide-page">
      <section className="page-heading"><div><p className="eyebrow">GROUP ROOM</p><h1>Tạo hoặc tham gia phòng</h1><p className="muted">Luồng bổ trợ dùng API room thật của Backend.</p></div></section>
      {error && <Alert type="error">{error}</Alert>}
      <section className="room-actions">
        <form className="panel" onSubmit={create}><h2>Tạo phòng</h2><label>Tên phòng<input value={roomName} onChange={(e) => setRoomName(e.target.value)} placeholder="Movie night" /></label><button className="button primary" disabled={loading}>Tạo phòng</button></form>
        <form className="panel" onSubmit={join}><h2>Tham gia phòng</h2><label>Mã phòng<input value={joinCode} onChange={(e) => setJoinCode(e.target.value.toUpperCase())} placeholder="ABC123" required /></label><button className="button secondary" disabled={loading}>Tham gia</button></form>
      </section>
      {room && <section className="room-lobby"><div className="room-code"><span>ROOM CODE</span><strong>{room.code}</strong></div><div><h2>{room.name || 'Phòng CineMatch'}</h2><p>Trạng thái: <strong>{room.status}</strong> · {room.members.length} thành viên</p></div><button className="button primary" onClick={ready} disabled={loading}>Đổi trạng thái Ready</button><div className="member-list">{room.members.map((member) => <div key={member.id}><span>{member.user?.email || `User #${member.user_id}`}</span><strong className={member.is_ready ? 'ready' : 'not-ready'}>{member.is_ready ? 'READY' : 'WAITING'}</strong></div>)}</div></section>}
    </main>
  )
}
