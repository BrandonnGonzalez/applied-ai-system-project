# VibeFinder 1.0 — Hybrid Music Recommender System

## Original Project

This project started as **Music Recommender Simulation** (CodePath AI110, Module 3). Its original goal was to represent songs and a user "taste profile" as structured data, design a rule-based scoring function to rank songs by how closely they matched user preferences, and reflect on how that mirrors real-world AI recommenders. The starter system scored songs purely by matching explicit features — genre, energy, tempo, valence — with no learned component and no safety guardrails.

---

## What It Does Now (and Why It Matters)

VibeFinder 1.0 is a **hybrid recommendation engine** that blends neural semantic understanding with explicit rule-based scoring to surface the top-K songs from a catalog for a given user. The system demonstrates a full production-style AI pipeline: structured input validation, two-stage hybrid inference, and output guardrails — all running locally with no API keys or cloud costs.

This project matters because it shows how even a small, transparent recommender reveals real design tensions that power every major music platform: how much weight to give genre vs. vibe, how to handle edge-case user profiles, and where a seemingly "smart" system can quietly produce biased results.

---

## Architecture Overview

```
User Preferences (dict)
        │
        ▼
┌───────────────────┐
│   Input Guard     │  ← guardrails-ai + Pydantic
│  (guards.py)      │     validates ranges, types, required fields
└────────┬──────────┘
         │ valid prefs
         ▼
┌──────────────────────────────────────────────┐
│              Hybrid Scorer (recommender.py)  │
│                                              │
│  ┌─────────────────────┐  weight: 60%        │
│  │  Semantic Similarity │ ← all-MiniLM-L6-v2 │
│  │  (embedder.py)       │   cosine sim        │
│  └─────────────────────┘                     │
│                                              │
│  ┌─────────────────────┐  weight: 40%        │
│  │  Rule-Based Scorer   │ ← genre, energy,   │
│  │  (recommender.py)    │   tempo, valence,  │
│  └─────────────────────┘   acousticness      │
│                                              │
│  final_score = 0.6 × semantic + 0.4 × rules  │
└────────────────────┬─────────────────────────┘
                     │ ranked (song, score, explanation) list
                     ▼
           ┌──────────────────┐
           │  Output Guard    │  ← clamps scores to [0,1]
           │  (guards.py)     │     fixes trivial explanations
           └────────┬─────────┘
                    │
                    ▼
          Top-K Recommendations
```

**Key components:**

| File | Responsibility |
|---|---|
| `src/main.py` | CLI runner; wires together all components and runs demo scenarios |
| `src/recommender.py` | `Song`, `UserProfile`, `Recommender` dataclasses + `recommend_songs()` functional API |
| `src/embedder.py` | Converts songs and user queries to natural-language descriptions; runs `all-MiniLM-L6-v2` for cosine similarity |
| `src/guards.py` | Input guard (Pydantic schema) + output guard (custom validators for score range and explanation quality) |
| `data/songs.csv` | 15-song catalog across 12 genres |

---

## Setup Instructions

**Prerequisites:** Python 3.9+

### 1. Clone and enter the project directory

```bash
cd ai110-module3show-musicrecommendersimulation-starter
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Mac / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> On first run, `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~22 MB) automatically. No API key is required.

### 4. Run the system

```bash
python -m src.main
```

### 5. Run the test suite

```bash
pytest
```

---

## Sample Interactions

The CLI runs four scenarios back-to-back. Below are representative inputs and the outputs produced.

---

### Example 1 — Happy Pop listener (valid request)

**Input preferences:**
```
favorite_genre : pop
favorite_mood  : happy
target_energy  : 0.8
target_tempo   : 120.0
target_valence : 0.85
likes_acoustic : False
```

**System output (top 5):**
```
#1  Sunrise City — Neon Echo  [pop / happy]
     Score : 0.776
     [Hybrid | all-MiniLM-L6-v2] semantic=0.86, rule=0.62 → score=0.78. Rule signals: genre match (+2.0), energy similarity (+0.96), tempo similarity (+0.49), valence similarity (+0.49).

#2  Gym Hero — Max Pulse  [pop / intense]
     Score : 0.742
     [Hybrid | all-MiniLM-L6-v2] semantic=0.84, rule=0.57 → score=0.73. Rule signals: genre match (+2.0), energy similarity (+0.87), tempo similarity (+0.44), valence similarity (+0.46).

#3  Rooftop Lights — Indigo Parade  [indie pop / happy]
     Score : 0.718
     [Hybrid | all-MiniLM-L6-v2] semantic=0.83, rule=0.52 → score=0.71. Rule signals: energy similarity (+0.94), tempo similarity (+0.48), valence similarity (+0.48).
```

**What this shows:** Genre match has the strongest pull in the rule-based layer, but the semantic layer picks up "indie pop" as close to "pop" and surfaces Rooftop Lights at #3 even without an exact genre match — something a pure rule-based system would miss.

---

### Example 2 — Input guard blocks bad data (energy = 2.5)

**Input preferences:**
```
favorite_genre : pop
favorite_mood  : happy
target_energy  : 2.5      ← INVALID (must be ≤ 1.0)
target_tempo   : 120.0
target_valence : 0.85
likes_acoustic : False
```

**System output:**
```
[INPUT GUARD BLOCKED]  Input guard blocked invalid user preferences —
target_energy: Input should be less than or equal to 1 [type=less_than_equal, ...].
Received: {'favorite_genre': 'pop', 'favorite_mood': 'happy', 'target_energy': 2.5, ...}
```

**What this shows:** The Pydantic-backed input guard rejects bad data before any model inference runs, keeping the pipeline safe and giving the caller a human-readable error with the exact failing field.

---

### Example 3 — Input guard blocks empty genre string

**Input preferences:**
```
favorite_genre : ""        ← INVALID (min length 1)
favorite_mood  : chill
target_energy  : 0.4
target_tempo   : 80.0
target_valence : 0.6
likes_acoustic : True
```

**System output:**
```
[INPUT GUARD BLOCKED]  Input guard blocked invalid user preferences —
favorite_genre: String should have at least 1 character [type=string_too_short, ...].
Received: {'favorite_genre': '', 'favorite_mood': 'chill', ...}
```

---

### Example 4 — Feature-removal experiment (mood scoring disabled)

During development, the `mood` feature was temporarily commented out of the rule-based scorer to measure its impact on rankings. Running this experiment with a chill/acoustic user profile showed that removing mood **did not reorder any songs** — it only reduced every song's raw score uniformly. This revealed that genre and energy dominate the rule-based component, making mood effectively redundant in the current weight configuration.

---

## Design Decisions

### Hybrid scoring (60% semantic / 40% rule-based)

A pure rule-based scorer can only match exact strings: `"indie pop" ≠ "pop"`, `"serene" ≠ "calm"`. Adding `all-MiniLM-L6-v2` — a 22 MB sentence-transformer model fine-tuned on 1 billion sentence pairs — lets the system understand those near-matches through cosine similarity on natural-language descriptions. The 60/40 split was chosen to give semantic understanding the majority voice while keeping the rule-based component as an interpretability anchor. A future iteration could learn this ratio from user feedback.

**Trade-off:** Semantic scoring adds ~50ms of CPU inference per request. For a 15-song catalog that is negligible; for a 1M-song catalog it would require a vector index (e.g., FAISS or Pinecone) rather than brute-force cosine comparison.

### Guardrails-ai on both ends

Input validation via Pydantic + `Guard.for_pydantic()` catches bad data before the model ever runs — no wasted inference on junk inputs. Output validation clamps scores to `[0, 1]` and replaces trivial explanations, preventing downstream consumers from receiving silent garbage. This two-gate pattern mirrors how production ML pipelines protect model inference.

**Trade-off:** Output fixing is silent by default (only raises a `warnings.warn`). A stricter system would surface these corrections to the caller so they can investigate data quality issues.

### Genre gets 2× weight in the rule layer

Genre is the most interpretable feature a user can state, and in practice it is the strongest signal for compatibility. Doubling its weight (2.0 vs. 0.5–1.0 for all other features) reflects that priority.

**Trade-off:** This creates a mild genre filter bubble. A user who asks for "lofi" will rarely see a great jazz track even if every other feature is a perfect match. Diversity-aware re-ranking (e.g., Maximum Marginal Relevance) could address this.

### Natural-language song descriptions for embedding

Rather than embedding raw feature vectors, `embedder.py` converts each song to a human-readable sentence (`"A lofi song called 'Midnight Coding' by LoRoom. The mood is chill. It is low-energy, slow-paced, emotionally neutral, and acoustic and organic."`). This grounds the embedding in the same semantic space as the user query, which is also expressed as a natural-language sentence. The result is a more stable and interpretable similarity signal than embedding numeric floats directly.

---

## Testing Summary

### What the tests cover

`tests/test_recommender.py` exercises the OOP API (`Recommender`, `Song`, `UserProfile`):

- `test_recommend_returns_songs_sorted_by_score` — verifies that a pop/happy user gets the pop/happy song ranked first, confirming the scoring direction.
- `test_explain_recommendation_returns_non_empty_string` — verifies that explanations are non-trivial strings, not empty or null.

### What worked

- The hybrid scorer consistently surfaced genre-matching songs first for standard profiles (HIGH_ENERGY_POP, CHILL_LOFI, DEEP_INTENSE_ROCK).
- The semantic layer correctly ranked "indie pop" songs near "pop" preferences without any explicit rule.
- Both guardrails caught every invalid input in the demo scenarios (out-of-range energy, negative tempo, empty genre string) and provided actionable error messages.
- The output guard's auto-fix for scores and explanations worked silently without crashing the recommendation pipeline.

### What didn't work as expected

- **Mood is nearly irrelevant.** Removing mood from the rule scorer (the feature-removal experiment) had zero effect on song rankings — only on raw scores. This means mood was being drowned out by genre and energy in every tested profile, which is a design flaw for users who care primarily about vibe over genre.
- **Niche genre profiles fail silently.** A user requesting `"lofi"` gets lofi songs ranked first, but a user requesting `"synthwave"` only gets one catalog match. The system does not warn about sparse catalog coverage.
- **Energy similarity is too forgiving.** Because `1 - abs(target - value)` is always positive, even a song with energy 0.1 gets a nonzero energy bonus for a user who wants energy 0.9. A threshold-based scorer would produce sharper separations.

### What I learned

- Small weight changes cascade in non-obvious ways. Doubling the genre weight to 2.0 makes it the single most powerful feature, which only became apparent by running the feature-removal experiment.
- Guardrails are most useful when they explain *why* they blocked something, not just *that* they blocked it. The Pydantic-backed error messages (field name + constraint + received value) made debugging significantly faster than a generic "validation failed" message.
- Tests prove logic correctness, not behavioral correctness. The two unit tests both pass, but they could not have caught the mood-redundancy issue — that required empirical experimentation with multiple user profiles.

---

## Reflection

Building VibeFinder forced me to confront a core tension in recommendation systems: **interpretability vs. flexibility**. Rule-based systems are transparent — you can read the weights and know exactly why a song ranked first — but they are brittle, unable to understand that "indie pop" is close to "pop" or that "serene" and "calm" are near-synonyms. Neural systems handle those fuzzy relationships naturally but operate as black boxes. The hybrid approach here is a first-principles attempt to get the best of both, and it works reasonably well within the catalog's constraints.

The bigger lesson was about bias. I expected the system to produce sensible results across diverse user profiles, and it mostly did — until I tested profiles with unusual combinations like high acoustic preference + high energy, or sad mood + high valence. Those edge cases exposed that the system was built implicitly around a "mainstream pop listener" mental model. The genre double-weight, the linear similarity functions, and the 15-song catalog all skew toward that center. Real recommenders face this at scale: the data and the weights encode whose tastes were considered "normal" when the system was designed. Human judgment is still essential for catching those assumptions — the model has no way to notice them itself.

---

## Model Card

See [model_card.md](model_card.md) for a full breakdown of intended use, data, strengths, limitations, evaluation methodology, and future work.

---

## Project Screenshots

**Phase 3 — Rule-based scoring baseline**

<img width="736" height="283" alt="Phase 3 screenshot" src="https://github.com/user-attachments/assets/e6461356-d445-4718-b29b-ac761fcda8cf" />

**Phase 4 — Hybrid scoring with guardrails**

<img width="759" height="263" alt="Phase 4 screenshot" src="https://github.com/user-attachments/assets/10aac16f-10d2-4929-9356-11c624bc5ce8" />

**Feature-removal experiment — mood scoring disabled**

<img width="737" height="308" alt="Feature removal experiment" src="https://github.com/user-attachments/assets/ba49a4c5-e273-4269-8910-c2e6654f9f22" />

Removing mood from the rule scorer did not change any song's rank — only lowered raw scores uniformly — confirming that genre and energy dominate the rule-based component.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Semantic model | `sentence-transformers/all-MiniLM-L6-v2` |
| Input/output validation | `guardrails-ai` + `pydantic` |
| Data | CSV (15 songs, 12 genres) |
| Tests | `pytest` |
| Numeric ops | `numpy` |
