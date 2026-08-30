# CineMatch Recommendation Contract v1

This document fixes the Group Recommendation data shape shared between
Sơn (Group Recommendation), Chúc (Backend) and Dương (Frontend).

## ID conventions

- `movie_id` is the original MovieLens ID, used when communicating with Data
  and AI.
- `database_id` is the Backend's internal key and must not be sent in place of
  `movie_id`.
- `movie_index` is used only inside the AI model's embeddings.

## Week 1 mock endpoint

```http
POST /recommend/mock
```

This endpoint serves only development and the week 1 demo. It does not call
the AI model, does not require real room data and does not write to the
database. The result is generated deterministically so that the Frontend and
the tests receive the same response on every run.

Request:

```json
{
  "room_id": 1,
  "strategy": "average",
  "top_k": 10
}
```

The three valid strategies:

```text
average
least_misery
average_without_misery
```

A sample response is in
`docs/examples/recommendation_response.json`.

## Meaning of the result fields

| Field | Meaning |
|---|---|
| `schema_version` | Version of the data contract |
| `room_id` | The room the recommendation is computed for |
| `strategy` | The aggregation strategy in use |
| `recommendations` | The Top-K list of movies |
| `movie_id` | Original MovieLens ID |
| `rank` | Ranking position, starting from 1 |
| `group_score` | Group score for that specific movie |
| `minimum_score` | Lowest member score |
| `disagreement` | Degree of disagreement between members |
| `member_scores` | Predicted score of each member |
| `misery_warning` | Whether any member is below the misery threshold |
| `explanations` | The reasons the movie is recommended |

`title`, `genres`, `poster_url` and `runtime_minutes` are looked up by the
Backend from the movie catalog. If MovieLens does not have a poster or runtime,
return `null`; do not fabricate incorrect metadata.

## Business rules

- `group_score`, `minimum_score` and `disagreement` are computed separately
  for each movie.
- Results are sorted by `rank` in ascending order.
- Every movie must have the members' scores in `member_scores`.
- Sơn is responsible for finalizing the formula, the misery threshold and the
  tie-break.
- Week 1 may use mock scores, but the request/response must keep the correct
  schema.
- `average_without_misery` excludes movies with `minimum_score < 2.0` in the
  mock data.
- The old draft using `top_movies` and `score` is still accepted by Pydantic
  during validation; the new standard output uses `recommendations` and
  `group_score`.
- `2.0` is the shared misery threshold. The Backend must use
  `DEFAULT_MISERY_THRESHOLD` from `cinematch.recommendation.group`, not
  declare a separate value.

## Model-to-Group Contract

### Model input

NCF and Hybrid NCF use:

```text
user_index
movie_index
```

Both indexes come from `data/processed/id_mappings.json`. The model never
sees `movie_id` or `database_id`; the caller converts before and after
scoring.

### Model output

For one room the model produces one score row per member, in a fixed member
order, with one prediction for every candidate movie:

```text
movie_ids         : (m_1, m_2, ..., m_N)          # original MovieLens IDs, unique, positive
member_score_rows : ((s_11, ..., s_1N),           # member 1
                     (s_21, ..., s_2N),           # member 2
                     ...)                         # 2 to 5 rows
```

Rules enforced by `prepare_group_candidate_scores(movie_ids,
member_score_rows)`:

- 2 to 5 member rows (`MIN_GROUP_SIZE`, `MAX_GROUP_SIZE`).
- Every row has exactly `N` finite numbers; NaN or infinity is rejected.
- Scores are clamped to the MovieLens rating scale `[1.0, 5.0]`
  (`MIN_SCORE`, `MAX_SCORE`) before aggregation.
- The output is `dict[movie_id, tuple[float, ...]]`, keyed by original
  `movie_id`, preserving member order.

The member order in `member_score_rows` must be the same order as
`member_user_ids` passed to the response builder; the Backend owns that
ordering.

### Aggregation

Implemented in `cinematch.recommendation.group` and applied per movie:

| Strategy | `group_score` | Effect |
|---|---|---|
| `average` | mean of member scores | Every member has equal weight; a high score can compensate a low one |
| `least_misery` | minimum member score | Protects the least satisfied member |
| `average_without_misery` | mean, but the movie is **excluded** when `min(scores) < misery_threshold` | Balanced default for the MVP |

Fields computed for every retained movie:

- `minimum_score = min(scores)`.
- `disagreement` = population standard deviation of the member scores
  (0.0 means all members agree).
- `misery_warning = minimum_score < misery_threshold` (informational; only
  `average_without_misery` actually removes the movie).

The misery threshold is `DEFAULT_MISERY_THRESHOLD = 2.0` on the 1–5 scale and
is the single source of truth for Backend, tests and documentation.

### Ranking and tie-break

`rank_group_candidates(candidate_scores, strategy, misery_threshold, top_k)`
sorts retained movies by:

1. Higher `group_score`.
2. Higher `minimum_score`.
3. Lower `disagreement`.
4. Lower `movie_id` (deterministic output).

and returns the first `top_k` items (default 10). If fewer than `top_k`
movies survive the filters, the shorter list is returned; the Backend must
surface this to the user rather than padding the list.

### Explanations

`build_group_recommendation_response` attaches `explanations` to every item.
Week 1 emits three template strings (strategy used, score summary, misery
status). From week 2 each item must carry at least two reasons that name a
real feature — genre match, satisfied constraint, member score or text
similarity — never invented facts.

## Model-to-Backend Artifact Contract

### Core response

`build_group_recommendation_response(room_id, member_user_ids,
candidate_scores, strategy, top_k=10, misery_threshold=2.0,
member_display_names=None)` returns a `GroupRecommendationResponse`:

```text
GroupRecommendationResponse
  schema_version : "1.0"
  room_id        : int
  strategy       : AggregationStrategy
  recommendations: tuple[GroupRecommendationItem, ...]

GroupRecommendationItem
  movie_id, rank, group_score, minimum_score, disagreement,
  member_scores : tuple[MemberPredictedScore(user_id, predicted_score, display_name), ...]
  misery_warning: bool
  explanations  : tuple[str, ...]
```

### Backend payload

`group_response_to_backend_payload(response, movie_metadata)` merges the core
response with display metadata supplied by the Backend, indexed by
`movie_id`:

```text
MovieResponseMetadata
  title           : str            # required, non-empty
  genres          : tuple[str, ...]# non-empty strings
  poster_url      : str | None
  runtime_minutes : int | None     # positive integer or None
```

The function raises if any recommended movie lacks metadata, so the Backend
must resolve the catalog before serializing. The result is a JSON-compatible
dictionary that validates against `GroupRecommendationOut` in
`app/schemas/run.py`, i.e. exactly the shape documented in "Meaning of the
result fields" above.

### Persistence

When a real run is stored (`recommendation_runs` / `recommendation_items`),
the Backend persists per item: `movie_id`, `rank`, `group_score`,
`minimum_score`, `disagreement`, the member scores and `explanations`, plus
per run: `model_version` (checkpoint identifier), `strategy`,
`misery_threshold` and the timestamp. Reproducing a stored run must be
possible from these fields alone.

### Versioning

`schema_version` is `"1.0"`. Adding optional fields keeps the version;
renaming or removing a field, or changing the score scale, requires bumping
it and updating `frontend/src/types/index.ts` in the same pull request.
