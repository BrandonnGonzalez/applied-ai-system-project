"""
Semantic embedding layer for the music recommender.

Model: sentence-transformers/all-MiniLM-L6-v2
  - Fine-tuned BERT model trained on 1B+ sentence pairs for semantic similarity
  - Runs locally on CPU, no API key or cost required
  - 22MB model size, ~50ms inference per batch on CPU

How it plugs into recommendations:
  Songs and user preferences are each converted to natural-language descriptions,
  embedded by the model, and scored with cosine similarity. This captures
  semantic relationships the rule-based scorer misses — e.g. "indie pop" ≈ "pop",
  "chill" ≈ "relaxed", "serene" ≈ "calm".
"""

from typing import Dict, List, Optional

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def song_to_description(song: Dict) -> str:
    energy = song["energy"]
    tempo = song["tempo_bpm"]
    valence = song["valence"]
    acousticness = song["acousticness"]

    energy_label = (
        "high-energy" if energy > 0.7 else
        "moderate-energy" if energy > 0.4 else
        "low-energy"
    )
    tempo_label = (
        "fast-paced" if tempo > 130 else
        "medium-paced" if tempo > 90 else
        "slow-paced"
    )
    valence_label = (
        "uplifting and positive" if valence > 0.7 else
        "emotionally neutral" if valence > 0.4 else
        "melancholic and dark"
    )
    acoustic_label = "acoustic and organic" if acousticness > 0.5 else "electronic and produced"

    return (
        f"A {song['genre']} song called '{song['title']}' by {song['artist']}. "
        f"The mood is {song['mood']}. "
        f"It is {energy_label}, {tempo_label}, {valence_label}, and {acoustic_label}."
    )


def user_prefs_to_query(user_prefs: Dict) -> str:
    energy = user_prefs.get("target_energy", 0.5)
    tempo = user_prefs.get("target_tempo", 100.0)
    valence = user_prefs.get("target_valence", 0.5)
    likes_acoustic = user_prefs.get("likes_acoustic", False)
    genre = user_prefs.get("favorite_genre", "any genre")
    mood = user_prefs.get("favorite_mood", "any mood")

    energy_label = (
        "high-energy" if energy > 0.7 else
        "moderate-energy" if energy > 0.4 else
        "low-energy"
    )
    tempo_label = (
        "fast-paced" if tempo > 130 else
        "medium-paced" if tempo > 90 else
        "slow-paced"
    )
    valence_label = (
        "uplifting and positive" if valence > 0.7 else
        "emotionally neutral" if valence > 0.4 else
        "melancholic and dark"
    )
    acoustic_label = "acoustic and organic" if likes_acoustic else "electronic and produced"

    return (
        f"I want a {genre} song with a {mood} mood. "
        f"I prefer {energy_label}, {tempo_label}, {valence_label}, and {acoustic_label} music."
    )


def compute_semantic_scores(user_prefs: Dict, songs: List[Dict]) -> Optional[List[float]]:
    """
    Returns a cosine-similarity score in [0, 1] for each song against the user query.
    Returns None if sentence-transformers is unavailable (graceful fallback to rule-based).
    """
    try:
        import numpy as np
        model = get_model()
    except Exception:
        return None

    user_query = user_prefs_to_query(user_prefs)
    song_texts = [song_to_description(song) for song in songs]

    all_texts = [user_query] + song_texts
    # normalize_embeddings=True means dot product == cosine similarity
    embeddings = model.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)

    user_vec = embeddings[0]
    song_vecs = embeddings[1:]

    # Cosine similarity is in [-1, 1]; scale to [0, 1]
    raw = song_vecs @ user_vec
    return ((raw + 1.0) / 2.0).tolist()


MODEL_NAME = _MODEL_NAME
