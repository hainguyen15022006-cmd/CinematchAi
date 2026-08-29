# Cinematch AI (AI NCF - HYBRID NCF)

## 1. Things I need to be able to explain

### Q1. NCF vs GMF — what does the MLP actually add?

GMF just multiplies the user embedding and item embedding together (element-wise), then one linear layer turns that into a score. It's basically MF with learnable weights.

NCF concatenates the two embeddings instead of multiplying, then runs them through a few MLP layers with ReLU. The point of the MLP is that multiplication can only capture a fixed kind of interaction, while stacked non-linear layers can pick up weirder patterns — like a user liking action movies only when there's also comedy, not action alone. GMF can't really represent that.

### Q2. Why does Dropout help with overfitting?

Every training step it randomly zeroes out some neurons, so the network can't lean on the same few neurons every time. It's kind of like training a bunch of slightly different smaller networks and averaging them.

At test time dropout is off, and the model ends up not just memorizing the training set — train/val loss stay closer together.

### Q3. Where do metadata and text get merged into Hybrid?

Right at the concatenation step, before the MLP. Normal NCF only concats `[user_emb, item_emb]`. Hybrid adds a side feature vector on top: `[user_emb, item_emb, side_feature]` — genres for now, text encoder output later.

Because it's merged at the input, the first MLP layer's input size has to match the new total length, which is the easiest place to get a shape mismatch if you forget to update it.

### Q4. Cold-start user vs unknown user — what's the actual difference?

**Cold-start user** = they're in the mapping, they have an embedding slot, they just have zero ratings so far. The embedding is basically still random since nothing trained it. Prediction will be bad, but nothing crashes.

**Unknown user** = not in the mapping at all, ID is out of range for the embedding table. That's a crash (index out of range), not a bad prediction — needs to be caught and mapped to some fallback "unknown" index before it even reaches the model.

So one is "we know about them but haven't learned anything yet," the other is "we literally have no slot for them."

---

## 2. Week 1 summary

Goal this week was just getting a working NCF baseline and understanding it well enough to explain, not chasing good numbers.

**Done:**
- NCF built with embeddings, MLP, forward pass — runs fine on a batch, no shape issues.
- Hybrid NCF built on top of it, concatenating a side feature vector into the input.
- Tests pass (2 for NCF, 1 for Hybrid).
- Ran a few epochs just to sanity-check training works: NCF loss went 8.11 → 6.17, Hybrid went 1.44 → 1.22. Hybrid starts lower/drops faster since it already has extra signal from the side features.

