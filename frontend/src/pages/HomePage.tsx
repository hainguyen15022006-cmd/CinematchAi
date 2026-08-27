import { Link } from 'react-router-dom'

export function HomePage() {
  return (
    <main className="page hero-page">
      <section className="hero-card">
        <p className="eyebrow">GROUP MOVIE RECOMMENDATION</p>
        <h1>Tìm bộ phim mà cả nhóm đều muốn xem.</h1>
        <p className="hero-copy">
          Chấm những phim bạn thích, vào phòng cùng bạn bè và xem Top 10 được tổng hợp theo điểm của cả nhóm.
        </p>
        <div className="hero-actions">
          <Link className="button primary" to="/register">Bắt đầu</Link>
          <Link className="button secondary" to="/recommendations">Xem mock Top 10</Link>
        </div>
      </section>
      <section className="feature-grid" aria-label="Các bước sử dụng">
        <article><span>01</span><h2>Tạo tài khoản</h2><p>Đăng ký và đăng nhập để nhận JWT từ Backend.</p></article>
        <article><span>02</span><h2>Chấm phim</h2><p>Frontend lấy catalog thật và gửi rating 1–5 qua API.</p></article>
        <article><span>03</span><h2>Chọn cùng nhau</h2><p>Top 10 hiển thị score, disagreement, member score và cảnh báo misery.</p></article>
      </section>
    </main>
  )
}
