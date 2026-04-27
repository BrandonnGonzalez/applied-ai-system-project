"""
Music Recommender — hybrid neural + rule-based CLI runner.

Recommendations are produced by blending two scoring signals:
  • Semantic similarity  (60%) — all-MiniLM-L6-v2 sentence-transformer model,
    fine-tuned on 1B+ sentence pairs, understands genre/mood relationships
    the rule-based scorer misses (e.g. "indie pop" ≈ "pop", "chill" ≈ "relaxed").
  • Rule-based score     (40%) — explicit feature matching on energy, tempo,
    valence, acousticness, and genre.
"""

import sys
import os

# Allow `from recommender import ...` to resolve to src/recommender.py
# when this file is run directly (python src/main.py) from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommender import load_songs, recommend_songs
from embedder import MODEL_NAME


def main() -> None:
    print(f"\nLoading semantic model: {MODEL_NAME} ...")
    songs = load_songs("data/songs.csv")

    # Starter example profile — swap in any values to explore the hybrid scorer
    user_prefs = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "target_tempo": 120,
        "target_valence": 0.85,
        "likes_acoustic": False,
    }

    print("\nUser preferences:")
    for key, value in user_prefs.items():
        print(f"  {key}: {value}")

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\n── Top 5 Recommendations ──────────────────────────────────────────────\n")
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"#{rank}  {song['title']} — {song['artist']}  [{song['genre']} / {song['mood']}]")
        print(f"    Score: {score:.3f}")
        print(f"    {explanation}")
        print()


if __name__ == "__main__":
    main()
