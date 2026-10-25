export interface User {
  id: number
  email: string
}

export interface UserProfile extends User {
  preferences_text?: string | null
}

export interface AuthToken {
  access_token: string
  token_type: string
}

export interface Movie {
  id: number
  movielens_id: number
  title: string
  genres?: string | null
  release_year?: number | null
  imdb_url?: string | null
}

export interface RatingPayload {
  movie_id: number
  rating: number
}

export interface Rating {
  id: number
  movie_id: number
  rating: number
  created_at: string
}

export type AggregationStrategy =
  | 'average'
  | 'least_misery'
  | 'average_without_misery'

export interface MemberScore {
  user_id: number
  display_name?: string | null
  predicted_score: number
}

export interface RecommendedMovie {
  movie_id: number
  rank: number
  title: string
  genres: string[]
  poster_url?: string | null
  runtime_minutes?: number | null
  group_score: number
  minimum_score: number
  disagreement: number
  member_scores: MemberScore[]
  misery_warning: boolean
  explanations: string[]
}

export interface RecommendationResponse {
  schema_version: string
  room_id: number
  strategy: AggregationStrategy
  recommendations: RecommendedMovie[]
}

export interface RoomMember {
  id: number
  user_id: number
  user?: User | null
  is_ready: boolean
  joined_at: string
}

export interface Room {
  id: number
  code: string
  host_id: number
  name?: string | null
  status: string
  constraints?: string | null
  created_at: string
  latest_run_id?: number | null
  members: RoomMember[]
}

/** Room settings the host picks before starting a run. */
export interface RoomConstraints {
  strategy?: AggregationStrategy
  top_k?: number
  misery_threshold?: number
}

export interface RunItem {
  id: number
  run_id: number
  movie_id: number
  rank: number
  ai_score?: number | null
  minimum_score?: number | null
  disagreement?: number | null
  misery_warning?: boolean | null
  member_scores: MemberScore[]
  explanations: string[]
  movie?: Movie | null
}

export interface RecommendationRun {
  id: number
  room_id: number
  status: string
  created_at: string
  strategy?: AggregationStrategy | null
  model_name?: string | null
  items: RunItem[]
}

export interface Vote {
  id: number
  run_id: number
  user_id: number
  movie_id: number
  vote_value: number
  created_at: string
  user?: User | null
}

export interface RunResult {
  run_id: number
  winner_movie?: Movie | null
  group_score?: number | null
  disagreement?: number | null
  votes: Vote[]
}

export interface RecommenderStatus {
  available: boolean
  model?: string | null
  detail: string
}
