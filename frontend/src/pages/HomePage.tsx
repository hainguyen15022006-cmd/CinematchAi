import { Link } from 'react-router-dom'

export function HomePage() {
  return (
    <main className="page hero-page">
      <section className="hero-card">
        <p className="eyebrow">GROUP MOVIE RECOMMENDATION</p>
        <h1>Find the movie everyone in the group wants to watch.</h1>
        <p className="hero-copy">
          Rate the movies you like, join a room with your friends, and see a Top 10 aggregated from the whole group's scores.
        </p>
        <div className="hero-actions">
          <Link className="button primary" to="/register">Get started</Link>
          <Link className="button secondary" to="/room">Open a group room</Link>
        </div>
      </section>
      <section className="feature-grid" aria-label="How it works">
        <article><span>01</span><h2>Create an account</h2><p>Register and sign in to receive a JWT from the Backend.</p></article>
        <article><span>02</span><h2>Rate movies</h2><p>The Frontend loads the real catalog and sends 1–5 ratings through the API.</p></article>
        <article><span>03</span><h2>Choose together</h2><p>The trained model scores each member, the group strategy ranks the movies, and the room votes on the winner.</p></article>
      </section>
    </main>
  )
}
