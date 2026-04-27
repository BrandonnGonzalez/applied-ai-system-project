import csv
import os
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Make the embedder importable whether we're run from the project root (via
# `python -m src.main`) or directly from the src/ directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedder import compute_semantic_scores, MODEL_NAME  # noqa: E402

# ── Hybrid scoring weights ────────────────────────────────────────────────────
# The final score blends neural semantic similarity with the explicit
# rule-based feature score. Semantic carries more weight because it captures
# relationships the rule-based scorer misses (e.g. "indie pop" ≈ "pop",
# "chill" ≈ "relaxed", mood nuance, etc.).
_SEMANTIC_WEIGHT = 0.6
_RULE_WEIGHT = 0.4
# Maximum achievable rule-based score (genre 2.0 + energy 1.0 + tempo 0.5
# + valence 0.5 + acoustic 1.0) used to normalize rule scores to [0, 1].
_MAX_RULE_SCORE = 5.0


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    target_tempo: float
    target_valence: float
    likes_acoustic: bool


# ── OOP Recommender ───────────────────────────────────────────────────────────

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py

    Uses a hybrid scoring strategy:
      final_score = 0.6 * semantic_similarity + 0.4 * normalised_rule_score

    The semantic component comes from all-MiniLM-L6-v2, a sentence-transformer
    fine-tuned on 1B+ sentence pairs. Songs and the user query are described in
    natural language, embedded, and compared with cosine similarity. This lets
    the system understand that "indie pop" ≈ "pop", "serene" ≈ "calm", etc.
    """

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        preferences = _user_profile_to_prefs(user)
        song_dicts = [_song_to_dict(s) for s in self.songs]

        semantic_scores = compute_semantic_scores(preferences, song_dicts)

        scored: List[Tuple[float, Song]] = []
        for i, song in enumerate(self.songs):
            rule_score, _ = score_song(preferences, song_dicts[i])
            rule_norm = min(rule_score / _MAX_RULE_SCORE, 1.0)

            if semantic_scores is not None:
                final = _SEMANTIC_WEIGHT * semantic_scores[i] + _RULE_WEIGHT * rule_norm
            else:
                final = rule_norm

            scored.append((final, song))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [song for _, song in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        preferences = _user_profile_to_prefs(user)
        song_data = _song_to_dict(song)
        rule_score, reasons = score_song(preferences, song_data)
        rule_norm = min(rule_score / _MAX_RULE_SCORE, 1.0)

        sem_scores = compute_semantic_scores(preferences, [song_data])
        if sem_scores is not None:
            sem_norm = sem_scores[0]
            final = _SEMANTIC_WEIGHT * sem_norm + _RULE_WEIGHT * rule_norm
            explanation = (
                f"Hybrid score: {final:.2f} "
                f"(semantic={sem_norm:.2f}, rule={rule_norm:.2f}). "
            )
        else:
            explanation = f"Rule-based score: {rule_score:.2f}. "

        if reasons:
            explanation += "Rule signals: " + ", ".join(reasons)
        else:
            explanation += "No strong rule-based matches found."

        return explanation


# ── Functional API ────────────────────────────────────────────────────────────

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            songs.append({
                "id": int(row["id"]),
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
                "energy": float(row["energy"]),
                "tempo_bpm": float(row["tempo_bpm"]),
                "valence": float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Rule-based score for a single song against user preferences.
    Required by recommend_songs() and the Recommender class.
    Returns (raw_score, list_of_reasons).
    """
    score = 0.0
    reasons: List[str] = []

    if song["genre"].lower() == user_prefs.get("favorite_genre", "").lower():
        score += 2.0
        reasons.append("genre match (+2.0)")

    # Temporarily remove mood as a feature to test ranking changes.
    # if song["mood"].lower() == user_prefs.get("favorite_mood", "").lower():
    #     score += 1.0
    #     reasons.append("mood match (+1.0)")

    energy_score = _similarity(user_prefs.get("target_energy", 0.0), song["energy"], 1.0) * 1.0
    if energy_score > 0:
        score += energy_score
        reasons.append(f"energy similarity (+{energy_score:.2f})")

    tempo_score = _similarity(user_prefs.get("target_tempo", 0.0), song["tempo_bpm"], 200.0) * 0.5
    if tempo_score > 0:
        score += tempo_score
        reasons.append(f"tempo similarity (+{tempo_score:.2f})")

    valence_score = _similarity(user_prefs.get("target_valence", 0.0), song["valence"], 1.0) * 0.5
    if valence_score > 0:
        score += valence_score
        reasons.append(f"valence similarity (+{valence_score:.2f})")

    if user_prefs.get("likes_acoustic", False):
        acoustic_score = song.get("acousticness", 0.0) * 1.0
        if acoustic_score > 0:
            score += acoustic_score
            reasons.append(f"acousticness bonus (+{acoustic_score:.2f})")

    return score, reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Hybrid recommender: combines neural semantic similarity with rule-based scoring.

    Scoring formula (per song):
      semantic_norm  = cosine_similarity(user_query_embed, song_embed)  → [0, 1]
      rule_norm      = rule_score / MAX_RULE_SCORE                       → [0, 1]
      final_score    = 0.6 * semantic_norm + 0.4 * rule_norm

    Falls back to pure rule-based scoring if sentence-transformers is unavailable.
    Required by src/main.py
    """
    semantic_scores = compute_semantic_scores(user_prefs, songs)

    ranked: List[Tuple[Dict, float, str]] = []
    for i, song in enumerate(songs):
        rule_score, reasons = score_song(user_prefs, song)
        rule_norm = min(rule_score / _MAX_RULE_SCORE, 1.0)

        if semantic_scores is not None:
            sem_norm = semantic_scores[i]
            final_score = _SEMANTIC_WEIGHT * sem_norm + _RULE_WEIGHT * rule_norm
            rule_summary = ", ".join(reasons) if reasons else "no rule-based matches"
            explanation = (
                f"[Hybrid | {MODEL_NAME}] "
                f"semantic={sem_norm:.2f}, rule={rule_norm:.2f} → score={final_score:.2f}. "
                f"Rule signals: {rule_summary}."
            )
        else:
            # Graceful fallback: sentence-transformers not installed
            final_score = rule_norm
            explanation = ", ".join(reasons) if reasons else "No strong matches found."

        ranked.append((song, final_score, explanation))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:k]


# ── Private helpers ───────────────────────────────────────────────────────────

def _similarity(target: float, value: float, scale: float) -> float:
    difference = abs(target - value)
    return max(0.0, 1.0 - (difference / scale))


def _song_to_dict(song: Song) -> Dict:
    return {
        "id": song.id,
        "title": song.title,
        "artist": song.artist,
        "genre": song.genre,
        "mood": song.mood,
        "energy": song.energy,
        "tempo_bpm": song.tempo_bpm,
        "valence": song.valence,
        "danceability": song.danceability,
        "acousticness": song.acousticness,
    }


def _user_profile_to_prefs(user: UserProfile) -> Dict:
    return {
        "favorite_genre": user.favorite_genre,
        "favorite_mood": user.favorite_mood,
        "target_energy": user.target_energy,
        "target_tempo": user.target_tempo,
        "target_valence": user.target_valence,
        "likes_acoustic": user.likes_acoustic,
    }


# ── Edge case user profiles (for bias/robustness testing) ─────────────────────

CONFLICTING_ENERGY_MOOD = {
    "favorite_genre": "pop",
    "favorite_mood": "sad",
    "target_energy": 0.9,
    "target_tempo": 130.0,
    "target_valence": 0.2,
    "likes_acoustic": False,
}

CONFLICTING_ACOUSTIC_ENERGY = {
    "favorite_genre": "folk",
    "favorite_mood": "chill",
    "target_energy": 0.95,
    "target_tempo": 120.0,
    "target_valence": 0.5,
    "likes_acoustic": True,
}

LOW_VALENCE_HAPPY_MOOD = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.5,
    "target_tempo": 100.0,
    "target_valence": 0.1,
    "likes_acoustic": False,
}

INVALID_GENRE = {
    "favorite_genre": "nonexistent_genre",
    "favorite_mood": "happy",
    "target_energy": 0.7,
    "target_tempo": 120.0,
    "target_valence": 0.8,
    "likes_acoustic": False,
}

EXTREME_HIGH_VALUES = {
    "favorite_genre": "rock",
    "favorite_mood": "intense",
    "target_energy": 1.0,
    "target_tempo": 200.0,
    "target_valence": 1.0,
    "likes_acoustic": False,
}

ALL_ZERO_VALUES = {
    "favorite_genre": "pop",
    "favorite_mood": "neutral",
    "target_energy": 0.0,
    "target_tempo": 0.0,
    "target_valence": 0.0,
    "likes_acoustic": False,
}

NEGATIVE_VALUES = {
    "favorite_genre": "electronic",
    "favorite_mood": "weird",
    "target_energy": -0.5,
    "target_tempo": -50.0,
    "target_valence": -0.2,
    "likes_acoustic": True,
}

EMPTY_STRINGS = {
    "favorite_genre": "",
    "favorite_mood": "",
    "target_energy": 0.5,
    "target_tempo": 100.0,
    "target_valence": 0.5,
    "likes_acoustic": False,
}

HIGH_ENERGY_POP = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.9,
    "target_tempo": 130.0,
    "target_valence": 0.9,
    "likes_acoustic": False,
}

CHILL_LOFI = {
    "favorite_genre": "lofi",
    "favorite_mood": "chill",
    "target_energy": 0.3,
    "target_tempo": 80.0,
    "target_valence": 0.6,
    "likes_acoustic": True,
}

DEEP_INTENSE_ROCK = {
    "favorite_genre": "rock",
    "favorite_mood": "intense",
    "target_energy": 0.95,
    "target_tempo": 150.0,
    "target_valence": 0.3,
    "likes_acoustic": False,
}
