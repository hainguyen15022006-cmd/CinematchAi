# CineMatch AI Architecture & Research Documentation (RQ2 Focus)

## 1. System Architecture Overview
CineMatch AI utilizes a **Hybrid Neural Collaborative Filtering (Hybrid NCF)** architecture that blends collaborative signals (User/Item ID Embeddings) with content-aware side features.

- **Input Dimension**: User Embedding (32) + Item Embedding (32) + Side Features (168)
- **Side Feature Vector Breakdown (168 dims)**:
  - `Genres` (19 dims): Multi-hot vector representing item categories.
  - `Year` (1 dim): Min-max normalized release year.
  - `History Profile` (19 dims): User historical preference distribution over genres.
  - `Text Embeddings` (128 dims): Item description representation generated via SentenceTransformer caching.
  - `Text Similarity` (1 dim): Cosine similarity score between user text preferences and item descriptions.
- **Output Layer**: Sigmoid activation scaled via $1.0 + 4.0 \cdot \sigma(x)$ producing bounded ratings in $[1.0, 5.0]$.

## 2. Research Question 2 (RQ2): Cold-Start Mitigation
To address RQ2 ("*How effectively do hybrid content features mitigate the cold-start problem for sparse/new users?*"):
- **Hybrid-Only Pathway**: When user historical interactions are negligible ($<5$ ratings), the model zero-out user ID embeddings and relies primarily on `history_profile`, `text_vector`, and `genres`.
- **Experimental Evidence**:
  - Evaluation results under cold-start simulations yielded **NDCG@5: 0.6149** and **Recall@5: 0.7052**.
  - Demonstrates robust ranking capability even in the total absence of collaborative filtering user historical interactions.

## 3. Pseudo-text Limitations & Item Text Representation
- **Pseudo-text Constraints**: In offline test scenarios without real-time API text streams, fixed pseudo-text embeddings generated via `TextEmbedderCache` serve as deterministic proxies.
- **Item-Side Bottlenecks**: High-dimensional text embeddings ($128$ dims) without fine-tuning may introduce noise, as observed in E4 validation loss compared to E3. Projection layers and caching mitigate CPU compute overhead while keeping latency under $30$ ms per candidate batch.