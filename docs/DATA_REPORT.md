# CineMatch Data Report

## 1. Dataset overview

CineMatch uses MovieLens 100K as the underlying data for
training and evaluating the recommendation models.

The rating data is read from `u.data` and has four columns:
`user_id`, `movie_id`, `rating` and `timestamp`.

| Attribute | Value |
|---|---:|
| Ratings | 100,000 |
| Users | 943 |
| Movies | 1,682 |
| Minimum rating | 1 |
| Maximum rating | 5 |
| Mean rating | 3.5299 |

## 2. Data quality

The pipeline checks the schema, missing values,
rating range, user IDs, movie IDs and timestamps.

Results:

- Missing values: 0.
- Ratings outside the 1-5 range: 0.
- Exact duplicate rows: 0.
- Duplicate user-movie rows: 0.
- Invalid user IDs: 0.
- Invalid movie IDs: 0.
- Invalid timestamps: 0.

The rating data meets the basic quality conditions
required to move on to the preprocessing step.

## 3. Rating distribution

| Rating | Count | Percentage |
|---:|---:|---:|
| 1 | 6,110 | 6.11% |
| 2 | 11,370 | 11.37% |
| 3 | 27,145 | 27.15% |
| 4 | 34,174 | 34.17% |
| 5 | 21,201 | 21.20% |

Rating 4 is the most frequent. The total number of ratings of 4
or higher is 55,375, or about 55.38%.

The data is skewed towards positive ratings.
CineMatch uses ratings of 4 or higher as positive
interactions in the ranking evaluation.

## 4. User interaction distribution

| Statistic | Ratings per user |
|---|---:|
| Minimum | 20 |
| Maximum | 737 |
| Mean | 106.04 |
| Median | 65 |

The mean being larger than the median shows that some users have
a very large number of ratings. However, every user has at least 20
ratings, which is enough to perform a per-user temporal split.

## 5. Movie interaction distribution

| Statistic | Ratings per movie |
|---|---:|
| Minimum | 1 |
| Maximum | 583 |
| Mean | 59.45 |
| Median | 27 |

The distribution of ratings per movie is long-tailed. Some movies
receive many ratings, while many movies have only a few interactions.
This can cause popularity bias and make the model learn poorly
for less popular movies.

## 6. Matrix sparsity

Number of possible user-movie pairs:

```text
943 × 1,682 = 1,586,126
```

Out of 1,586,126 possible interactions, the dataset contains only
100,000 ratings. The density is 6.3047% and the sparsity is 93.6953%.
This is a normal characteristic of recommendation data and is
the reason the MF, GMF and NCF models use embeddings.

## 7. Movie metadata quality

The `u.item` catalog contains 1,682 movies, and every movie ID that appears
in the ratings exists in the catalog.

| Check | Result |
|---|---:|
| Missing title | 0 |
| Missing release date | 1 |
| Missing IMDb URL | 3 |
| Movies with the `unknown` genre | 2 |
| Movies with more than one genre | 849 |
| Mean number of genres per movie | 1.72 |
| Ratings before the release date recorded in the catalog | 231 |
| Movies with a temporal inconsistency | 24 |

The IMDb URL is display metadata only and is not used for
training. Movies with a missing release date or with the `unknown`
genre are kept because they have valid rating interactions.

The release dates in `u.item` are not fully consistent with the
rating timestamps. Therefore, the pipeline does not use the release
date as a condition for removing ratings or filtering movies. The temporal
split is based only on the rating timestamps.

In the processed catalog, `release_date` is parsed to a date
type, `release_year` is a nullable integer and
`release_date_missing` records which entries are missing the date. The
release date is currently only secondary metadata, not a primary input
of MF, GMF or NCF.

## 8. Data processing policy

- Do not modify the files in `data/raw`.
- Do not remove ratings based on the release date.
- Do not fill in missing release dates or IMDb URLs.
- Keep all 19 genre columns, including `unknown`.
- Use the rating timestamp for the per-user temporal split.
- Store the transformed data in `data/processed`.

These decisions keep the pipeline reproducible and avoid
introducing unverified assumptions into the original data.

## 9. Per-user temporal split

After mapping user and movie IDs, each user's ratings are
sorted in ascending order of `timestamp`. `movie_id` is used as a
secondary key when two ratings share the same timestamp so that the
result is always reproducible.

Each user is split approximately by the ratio:

- The oldest 80% of interactions for train.
- The next 10% for validation.
- The newest 10% for test.

Because each user's number of interactions is an integer and is not
always divisible according to the 80/10/10 ratio, the pipeline uses the
largest remainder method to allocate the remainder. The result over the
whole of MovieLens 100K is:

| Partition | Rows | Percentage |
|---|---:|---:|
| Train | 80,014 | 80.014% |
| Validation | 10,132 | 10.132% |
| Test | 9,854 | 9.854% |
| Total | 100,000 | 100% |

All three partitions contain all 943 users. The pipeline checks that each
interaction appears exactly once and is neither lost nor
duplicated across partitions. Movie release dates play no part
in the data split.

## 10. Post-split audit and cold-start

Post-split audit confirms that the three partitions contain
100,000 interactions in total and that no interaction appears
in more than one partition.

| Check | Train | Validation | Test |
|---|---:|---:|---:|
| Rows | 80,014 | 10,132 | 9,854 |
| Users | 943 | 943 | 943 |
| Movies | 1,611 | 1,323 | 1,377 |
| Positive rate | 57.2750% | 48.1938% | 47.3310% |

Integrity results:

- User cold-start count: 0.
- Validation item cold-start: 33 movies and 36 interactions.
- Test item cold-start: 45 movies and 52 interactions.
- Cross-partition interaction overlap: 0.
- Train-validation temporal violations: 0.
- Validation-test temporal violations: 0.

Cold-start interactions are retained because they represent a
real limitation of collaborative filtering rather than corrupt
data. Evaluation must report metrics on the complete temporal
test set and may additionally report warm-start metrics using
only movies observed in the training partition.

All compared recommendation models must use the same evaluation
protocol and candidate construction rules.

## 11. Data handoff and known limitations

The shared configuration source is `configs/cinematch.yaml`. The pipeline
verifies that the dataset has exactly 100,000 ratings, 943 users and
1,682 movies before producing model-ready data.

The Data handoff to AI consists of:

- `train.csv`, `validation.csv` and `test.csv`.
- `movies.csv` with metadata and 19 genre columns.
- `id_mappings.json` for stable ID encoding/decoding.
- `split_audit.json` recording the post-split check results.
- `data_manifest.json` recording versions, calculated counts and artifact
  checksums.
- `catalog.json`, `seen_items.json`, `positive_test_items.json` and
  `evaluation_data_summary.json` as the reproducible input handoff to the
  Evaluation owner.
- A loader with an explicit schema in `cinematch.data.io`.

The current data version is `ml100k-temporal-v1`. Hybrid side features follow
contract `hybrid-v1-167` in this fixed order: 19 movie genres, one normalized
release-year value, 19 train-only user genre-history values and a
128-dimensional preference-text vector. The real feature values are produced
in the next feature-engineering stage; the manifest fixes their interface
before model training begins.

MovieLens 100K does not provide modern posters or movie
runtimes. Therefore, the posters and runtime constraints of the interface must
be supplemented from another metadata source; values must not be fabricated
in the training data. Free-text user preferences are onboarding data
collected by the system later and are not available in MovieLens 100K.

The pipeline does not yet generate negative samples. Negative sampling
depends on the objective and the evaluation protocol, so it must be agreed
between the AI and Evaluation owners, while using the same candidate
set for MF, GMF and NCF.

Given the limitations above, the current data is sufficient to train and
compare the baseline, MF, GMF and NCF on explicit ratings, while also
providing genre features for Hybrid NCF.

## 12. Evaluation eligibility handoff

Using a positive threshold of 4.0, 836 of the 943 users have at least one
positive item in the test partition. The other 107 users remain in the handoff
with an empty positive list and must be reported as skipped for ranking
metrics. Seen items are the union of train and validation interactions.

The generated evaluation artifacts use model indices, retain all 1,682 catalog
movies and distinguish 1,611 movies observed during training from cold-start
items. Full-catalog and warm-start metrics can therefore be reported without
changing the original temporal split. The artifact contract is documented in
`docs/EVALUATION_DATA_HANDOFF.md`.
