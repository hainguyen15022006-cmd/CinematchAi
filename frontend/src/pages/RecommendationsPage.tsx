import { FormEvent, useState } from 'react'
import { Alert } from '../components/Alert'
import { LoadingState } from '../components/LoadingState'
import { api, ApiError } from '../services/api'
import type { AggregationStrategy, RecommendationResponse } from '../types'

const labels: Record<AggregationStrategy, string> = {
  average: 'Average',
  least_misery: 'Least Misery',
  average_without_misery: 'Average Without Misery',
}

export function RecommendationsPage() {
  const [roomId, setRoomId] = useState(1)
  const [strategy, setStrategy] = useState<AggregationStrategy>('average')
  const [result, setResult] = useState<RecommendationResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function load(event?: FormEvent) {
    event?.preventDefault()
    setError('')
    setLoading(true)
    try {
      setResult(await api.recommendations(roomId, strategy, 10))
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Không lấy được recommendation.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="page wide-page">
      <section className="page-heading recommendation-heading">
        <div><p className="eyebrow">GROUP RECOMMENDATION · CONTRACT V1</p><h1>Top 10 cho cả nhóm</h1><p className="muted">Mock endpoint tuần 1. Không cần đăng nhập và giữ đúng schema đã chốt.</p></div>
        <form className="recommend-control" onSubmit={load}>
          <label>Room ID<input type="number" min="1" value={roomId} onChange={(e) => setRoomId(Math.max(1, Number(e.target.value)))} /></label>
          <label>Chiến lược<select value={strategy} onChange={(e) => setStrategy(e.target.value as AggregationStrategy)}><option value="average">Average</option><option value="least_misery">Least Misery</option><option value="average_without_misery">Average Without Misery</option></select></label>
          <button className="button primary" disabled={loading}>Tạo Top 10</button>
        </form>
      </section>
      {error && <Alert type="error">{error}</Alert>}
      {loading && <LoadingState label="Đang tổng hợp điểm nhóm..." />}
      {!loading && !result && <section className="empty-state"><div className="empty-icon">10</div><h2>Chưa có danh sách đề xuất</h2><p>Chọn strategy rồi bấm “Tạo Top 10”.</p></section>}
      {result && (
        <>
          <div className="result-meta"><span>Schema <strong>{result.schema_version}</strong></span><span>Room <strong>#{result.room_id}</strong></span><span>Strategy <strong>{labels[result.strategy]}</strong></span><span><strong>{result.recommendations.length}</strong> phim</span></div>
          <section className="recommend-list">
            {result.recommendations.map((movie) => (
              <article className="recommend-card" key={`${movie.rank}-${movie.movie_id}`}>
                <div className="rank-badge">#{movie.rank}</div>
                <div className="recommend-main">
                  <div className="recommend-title-row"><div><h2>{movie.title}</h2><p>{movie.genres.join(' · ') || 'Chưa có thể loại'}{movie.runtime_minutes ? ` · ${movie.runtime_minutes} phút` : ''}</p></div>{movie.misery_warning && <span className="warning-badge">⚠ Misery warning</span>}</div>
                  <div className="score-grid"><div><span>Group score</span><strong>{movie.group_score.toFixed(2)}</strong></div><div><span>Minimum</span><strong>{movie.minimum_score.toFixed(2)}</strong></div><div><span>Disagreement</span><strong>{movie.disagreement.toFixed(2)}</strong></div></div>
                  <div className="recommend-details">
                    <div><h3>Điểm thành viên</h3><div className="member-scores">{movie.member_scores.map((member) => <span key={member.user_id}><b>{member.display_name || `User ${member.user_id}`}</b><strong>{member.predicted_score.toFixed(1)}</strong></span>)}</div></div>
                    <div><h3>Giải thích</h3><ul>{movie.explanations.map((explanation, index) => <li key={index}>{explanation}</li>)}</ul></div>
                  </div>
                </div>
              </article>
            ))}
          </section>
        </>
      )}
    </main>
  )
}
