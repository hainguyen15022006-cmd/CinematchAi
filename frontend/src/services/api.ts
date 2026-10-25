import type {
  AggregationStrategy,
  AuthToken,
  Movie,
  Rating,
  RecommendationResponse,
  RecommendationRun,
  RecommenderStatus,
  Room,
  RoomConstraints,
  RoomMember,
  RunItem,
  RunResult,
  User,
  UserProfile,
  Vote,
} from '../types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const TOKEN_KEY = 'cinematch_access_token'

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

async function request<T>(path: string, init: RequestInit = {}, authenticated = false): Promise<T> {
  const headers = new Headers(init.headers)
  headers.set('Content-Type', 'application/json')

  if (authenticated) {
    const token = getToken()
    if (!token) throw new ApiError('You are not signed in.', 401)
    headers.set('Authorization', `Bearer ${token}`)
  }

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers })
  } catch {
    throw new ApiError('Could not connect to the Backend. Check that the server is running on port 8000.', 0)
  }

  if (!response.ok) {
    let message = `Request failed (${response.status}).`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) message = body.detail
    } catch {
      // Backend did not return a JSON error body.
    }
    throw new ApiError(message, response.status)
  }

  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string }>('/health'),

  register: (email: string, password: string) =>
    request<User>('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  login: (email: string, password: string) =>
    request<AuthToken>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  me: () => request<UserProfile>('/users/me', {}, true),

  updatePreferences: (preferences_text: string) =>
    request<UserProfile>(
      '/users/me/preferences',
      { method: 'PUT', body: JSON.stringify({ preferences_text }) },
      true,
    ),

  movies: (limit = 50, skip = 0) =>
    request<Movie[]>(`/movies?limit=${limit}&skip=${skip}`),

  rateMovie: (movie_id: number, rating: number) =>
    request<Rating>(
      '/ratings',
      { method: 'POST', body: JSON.stringify({ movie_id, rating }) },
      true,
    ),

  myRatings: () => request<Rating[]>('/ratings', {}, true),

  /** Week-one demo endpoint with fixed data; needs no sign-in. */
  mockRecommendations: (
    room_id: number,
    strategy: AggregationStrategy,
    top_k = 10,
  ) =>
    request<RecommendationResponse>('/recommend/mock', {
      method: 'POST',
      body: JSON.stringify({ room_id, strategy, top_k }),
    }),

  /** Tells whether the Backend can score with the trained model. */
  recommenderStatus: () => request<RecommenderStatus>('/recommend/status'),

  createRoom: (name?: string) =>
    request<Room>(
      '/rooms',
      { method: 'POST', body: JSON.stringify({ name: name?.trim() || null }) },
      true,
    ),

  getRoom: (code: string) => request<Room>(`/rooms/${code.toUpperCase()}`, {}, true),

  joinRoom: (code: string) =>
    request<Room>(`/rooms/${code.toUpperCase()}/join`, { method: 'POST' }, true),

  toggleReady: (roomId: number) =>
    request<RoomMember>(`/rooms/${roomId}/ready`, { method: 'POST' }, true),

  /** Host-only: store the strategy and Top-K used by the next run. */
  updateRoomConstraints: (roomId: number, constraints: RoomConstraints) =>
    request<Room>(
      `/rooms/${roomId}/constraints`,
      {
        method: 'PUT',
        body: JSON.stringify({ constraints: JSON.stringify(constraints) }),
      },
      true,
    ),

  /** Host-only: score every member and open voting. */
  triggerRecommendation: (roomId: number) =>
    request<RecommendationRun>(
      `/rooms/${roomId}/recommend`,
      { method: 'POST' },
      true,
    ),

  runItems: (runId: number) =>
    request<RunItem[]>(`/runs/${runId}/items`, {}, true),

  /** The stored Top-K of a run, in the shared recommendation contract. */
  runRecommendations: (runId: number) =>
    request<RecommendationResponse>(
      `/runs/${runId}/recommendations`,
      {},
      true,
    ),

  castVote: (runId: number, movie_id: number, vote_value = 1) =>
    request<Vote>(
      `/runs/${runId}/votes`,
      { method: 'POST', body: JSON.stringify({ movie_id, vote_value }) },
      true,
    ),

  /** Host-only: close voting and pick the winner. */
  finalizeRun: (runId: number) =>
    request<RunResult>(`/runs/${runId}/finalize`, { method: 'POST' }, true),

  runResult: (runId: number) =>
    request<RunResult>(`/runs/${runId}/result`, {}, true),
}

export { API_BASE_URL }
