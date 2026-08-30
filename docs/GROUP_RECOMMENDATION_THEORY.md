# Group Recommendation Theory

## 1. Objective

The Group Recommendation module takes each member's individual
predicted score for each movie, then aggregates them into a single
group score to rank movies for the whole group.

This module does not train a Deep Learning model. Individual
scores are provided by Most Popular, MF, GMF, NCF or Hybrid
NCF.

In week 1, predicted scores use a scale from 1 to 5.

## 2. Illustrative input data

Suppose the group has four members and three movies:

| Movie | Member 1 | Member 2 | Member 3 | Member 4 |
|---|---:|---:|---:|---:|
| A | 5.0 | 5.0 | 5.0 | 1.5 |
| B | 4.0 | 4.0 | 4.0 | 4.0 |
| C | 4.5 | 3.5 | 4.0 | 3.0 |

Each row is a movie. The Member columns are the scores the model
predicts for each member.

## 3. Average

Average computes the arithmetic mean of all members' scores:

GroupScore(i) = sum(score(u, i)) / number_of_members

### Worked by hand

Movie A:

Average(A) = (5.0 + 5.0 + 5.0 + 1.5) / 4
           = 16.5 / 4
           = 4.125

Movie B:

Average(B) = (4.0 + 4.0 + 4.0 + 4.0) / 4
           = 4.0

Movie C:

Average(C) = (4.5 + 3.5 + 4.0 + 3.0) / 4
           = 3.75

Ranking by Average:

1. Movie A: 4.125
2. Movie B: 4.0
3. Movie C: 3.75

### Remarks

Average optimizes mean satisfaction. However, Movie A
is ranked first even though Member 4 only has a score of 1.5. Thus, high
scores from the majority can mask strong objection from one member.

## 4. Least Misery

Least Misery uses the lowest score among the members:

GroupScore(i) = min(score(u, i))

### Worked by hand

LeastMisery(A) = min(5.0, 5.0, 5.0, 1.5) = 1.5

LeastMisery(B) = min(4.0, 4.0, 4.0, 4.0) = 4.0

LeastMisery(C) = min(4.5, 3.5, 4.0, 3.0) = 3.0

Ranking by Least Misery:

1. Movie B: 4.0
2. Movie C: 3.0
3. Movie A: 1.5

### Remarks

Least Misery protects the least satisfied member. A single
member can almost veto a movie by having a very low
score. The drawback is that this strategy can be overly conservative and
fail to take advantage of the majority's high satisfaction.

## 5. Average Without Misery

Average Without Misery proceeds in two stages:

1. Remove movies whose minimum score is below the misery threshold.
2. Compute the Average for the remaining movies.

In this example:

misery_threshold = 2.0

Rules:

- minimum_score < 2.0: the movie is removed.
- minimum_score >= 2.0: the movie is kept.

### Worked by hand

Movie A:

minimum_score = 1.5

Since 1.5 < 2.0, Movie A is removed.

Movie B:

minimum_score = 4.0
average_score = 4.0

Movie B is kept with a GroupScore of 4.0.

Movie C:

minimum_score = 3.0
average_score = 3.75

Movie C is kept with a GroupScore of 3.75.

Ranking by Average Without Misery:

1. Movie B: 4.0
2. Movie C: 3.75

Movie A does not appear because it violates the misery threshold.

### Remarks

Average Without Misery balances Average and Least
Misery. The minimum score is used as a filtering condition,
but movies that pass the condition are still ranked by
Average.

## 6. Minimum score

The minimum score is the lowest predicted score in the group:

MinimumScore(i) = min(score(u, i))

Results:

| Movie | Minimum score |
|---|---:|
| A | 1.5 |
| B | 4.0 |
| C | 3.0 |

The minimum score helps explain the satisfaction level of the least
satisfied member.

## 7. Disagreement

Disagreement measures the dispersion of scores among members. In
week 1, the module uses the population standard deviation:

Disagreement(i) =
sqrt(sum((score(u, i) - average(i))^2) / number_of_members)

A small value indicates high consensus. A large value indicates
that members have differing opinions.

### Movie A

Average(A) = 4.125

Squared differences:

- (5.0 - 4.125)^2 = 0.765625
- (5.0 - 4.125)^2 = 0.765625
- (5.0 - 4.125)^2 = 0.765625
- (1.5 - 4.125)^2 = 6.890625

Sum = 9.1875

Variance = 9.1875 / 4 = 2.296875

Disagreement(A) = sqrt(2.296875) = 1.5155

### Movie B

All members have a score of 4.0, so:

Disagreement(B) = 0.0

### Movie C

Average(C) = 3.75

Squared differences:

- (4.5 - 3.75)^2 = 0.5625
- (3.5 - 3.75)^2 = 0.0625
- (4.0 - 3.75)^2 = 0.0625
- (3.0 - 3.75)^2 = 0.5625

Sum = 1.25

Variance = 1.25 / 4 = 0.3125

Disagreement(C) = sqrt(0.3125) = 0.5590

### Results

| Movie | Average | Minimum | Disagreement |
|---|---:|---:|---:|
| A | 4.125 | 1.5 | 1.5155 |
| B | 4.000 | 4.0 | 0.0000 |
| C | 3.750 | 3.0 | 0.5590 |

Movie B has the highest consensus. Movie A has the highest
disagreement.

## 8. Distinguishing the two thresholds

The positive rating threshold and the misery threshold do not serve the
same purpose.

### Positive rating threshold

positive_rating_threshold = 4.0

Used in evaluation. A movie the user rated 4 or higher
is considered relevant.

### Misery threshold

misery_threshold = 2.0

Used in Average Without Misery. A movie is removed if any
member receives a predicted score below 2.0.

The two thresholds must not be used interchangeably.

## 9. Planned tie-break rules

When two movies have the same GroupScore, the system is planned to prioritize:

1. Higher minimum score.
2. Lower disagreement.
3. Smaller Movie ID so that results are reproducible.

These rules will be tested before Backend integration.

## 10. Comparison of strategies

| Strategy | Computation | Advantages | Limitations |
|---|---|---|---|
| Average | Mean of scores | Optimizes mean satisfaction | May ignore objectors |
| Least Misery | Lowest score | Protects the least satisfied member | May be overly conservative |
| Average Without Misery | Filter by minimum, then take the average | Balances both objectives | Depends on the misery threshold |

## 11. References

- Stratigi et al., "Sequential group recommendations based on
  satisfaction and disagreement scores", Journal of Intelligent
  Information Systems.
  https://link.springer.com/article/10.1007/s10844-021-00652-x