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
  members: RoomMember[]
}
