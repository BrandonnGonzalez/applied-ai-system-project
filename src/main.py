"""
Music Recommender — hybrid neural + rule-based CLI runner with guardrails.

Every request flows through two guardrails-ai checkpoints:
  • Input guard   — validates user preferences before any inference runs.
                    Raises InputValidationError immediately on bad data.
  • Output guard  — validates every recommendation's score and explanation.
                    Auto-fixes out-of-range scores; pads trivial explanations.

Scoring uses all-MiniLM-L6-v2 (60% weight) + rule-based features (40% weight).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommender import load_songs, recommend_songs
from guards import InputValidationError
from embedder import MODEL_NAME


def _print_separator(label: str = "") -> None:
    width = 70
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * pad}")
    else:
        print("─" * width)


def run_recommendations(user_prefs: dict, songs: list, label: str) -> None:
    """Attempt a recommendation run; display results or the guard error."""
    _print_separator(label)
    print("User preferences:")
    for key, value in user_prefs.items():
        print(f"  {key}: {value}")
    print()

    try:
        recommendations = recommend_songs(user_prefs, songs, k=5)
        print(f"Top {len(recommendations)} recommendations:\n")
        for rank, (song, score, explanation) in enumerate(recommendations, start=1):
            print(
                f"  #{rank}  {song['title']} — {song['artist']}"
                f"  [{song['genre']} / {song['mood']}]"
            )
            print(f"       Score : {score:.3f}")
            print(f"       {explanation}")
            print()
    except InputValidationError as exc:
        print(f"  [INPUT GUARD BLOCKED]  {exc}\n")


def main() -> None:
    print(f"\nSemantic model : {MODEL_NAME}")
    print("Guards         : guardrails-ai input + output validation\n")

    songs = load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs from catalog.")

    # ── Valid request — full pipeline runs ────────────────────────────────────
    valid_prefs = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "target_tempo": 120.0,
        "target_valence": 0.85,
        "likes_acoustic": False,
    }
    run_recommendations(valid_prefs, songs, label="Valid request")

    # ── Invalid: energy out of range [0, 1] ───────────────────────────────────
    bad_energy = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 2.5,        # INVALID — must be ≤ 1.0
        "target_tempo": 120.0,
        "target_valence": 0.85,
        "likes_acoustic": False,
    }
    run_recommendations(bad_energy, songs, label="Bad energy (2.5 > 1.0)")

    # ── Invalid: empty genre string ───────────────────────────────────────────
    bad_genre = {
        "favorite_genre": "",        # INVALID — must be at least 1 char
        "favorite_mood": "chill",
        "target_energy": 0.4,
        "target_tempo": 80.0,
        "target_valence": 0.6,
        "likes_acoustic": True,
    }
    run_recommendations(bad_genre, songs, label="Empty genre string")

    # ── Invalid: tempo below minimum (20 BPM) ────────────────────────────────
    bad_tempo = {
        "favorite_genre": "ambient",
        "favorite_mood": "serene",
        "target_energy": 0.2,
        "target_tempo": -10.0,       # INVALID — negative tempo
        "target_valence": 0.7,
        "likes_acoustic": True,
    }
    run_recommendations(bad_tempo, songs, label="Negative tempo (-10 BPM)")

    _print_separator()


if __name__ == "__main__":
    main()
