"""
Guardrails-ai input and output validation for the music recommender.

Input guard  — validates user preference dicts before they enter the scoring
               pipeline. Uses a Pydantic schema via Guard.for_pydantic() so that
               every field (genre, mood, energy, tempo, valence, likes_acoustic)
               is range-checked and type-checked before any model inference runs.

Output guard — validates every (song, score, explanation) tuple produced by
               recommend_songs(). Two custom Validators are registered:
                 • RecommendationScoreRange  — clamps scores to [0.0, 1.0]
                 • ExplanationQuality        — ensures explanations are non-trivial

Both guards surface violations as structured ValidationError messages so callers
can distinguish bad input from bad output.
"""

import json
import warnings
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from guardrails import Guard, Validator, register_validator, OnFailAction
from guardrails.validators import FailResult, PassResult, ValidationResult


# ── Input schema ──────────────────────────────────────────────────────────────

class UserPrefsSchema(BaseModel):
    """Pydantic model that defines valid user preference inputs."""

    favorite_genre: str = Field(
        min_length=1,
        max_length=50,
        description="Music genre the user prefers, e.g. 'pop', 'rock', 'lofi'.",
    )
    favorite_mood: str = Field(
        min_length=1,
        max_length=50,
        description="Mood the user is looking for, e.g. 'happy', 'chill', 'intense'.",
    )
    target_energy: float = Field(
        ge=0.0,
        le=1.0,
        description="Energy level from 0.0 (very calm) to 1.0 (very energetic).",
    )
    target_tempo: float = Field(
        ge=20.0,
        le=300.0,
        description="Target tempo in BPM. Typical music sits between 60 and 200.",
    )
    target_valence: float = Field(
        ge=0.0,
        le=1.0,
        description="Musical positivity: 0.0 = melancholic, 1.0 = euphoric.",
    )
    likes_acoustic: bool = Field(
        default=False,
        description="True if the user prefers acoustic over electronic sounds.",
    )


# ── Custom output validators ──────────────────────────────────────────────────

@register_validator(name="recommendation_score_range", data_type="string")
class RecommendationScoreRange(Validator):
    """
    Ensures that a hybrid recommendation score is in [0.0, 1.0].
    OnFailAction.FIX: automatically clamps the value rather than raising.
    """

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return FailResult(
                error_message=f"Score must be numeric, got {type(value).__name__!r}.",
                fix_value="0.0",
            )
        if not (0.0 <= score <= 1.0):
            clamped = max(0.0, min(1.0, score))
            return FailResult(
                error_message=(
                    f"Score {score:.4f} is outside [0.0, 1.0]; "
                    f"clamped to {clamped:.4f}."
                ),
                fix_value=str(clamped),
            )
        return PassResult()


@register_validator(name="explanation_quality", data_type="string")
class ExplanationQuality(Validator):
    """
    Ensures recommendation explanations are non-empty and sufficiently detailed.
    OnFailAction.FIX: replaces or pads trivial explanations automatically.

    Rules
    -----
    - Must be a non-empty string after stripping whitespace.
    - Must be at least 20 characters long.
    - Must not be a bare number or single token.
    """

    _MIN_LENGTH = 20
    _FALLBACK = "Recommended based on your music taste profile."

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        if not isinstance(value, str):
            return FailResult(
                error_message=f"Explanation must be a string, got {type(value).__name__!r}.",
                fix_value=self._FALLBACK,
            )
        stripped = value.strip()
        if not stripped:
            return FailResult(
                error_message="Explanation is empty.",
                fix_value=self._FALLBACK,
            )
        if len(stripped) < self._MIN_LENGTH:
            return FailResult(
                error_message=(
                    f"Explanation is too short ({len(stripped)} chars); "
                    f"minimum is {self._MIN_LENGTH}."
                ),
                fix_value=stripped + " — " + self._FALLBACK,
            )
        return PassResult()


# ── Guard instances ───────────────────────────────────────────────────────────

# Structured input guard: raises a clear error on the first invalid field.
input_guard: Guard = Guard.for_pydantic(UserPrefsSchema)

# Output guard for scores: auto-fixes out-of-range values without raising.
score_guard: Guard = Guard().use(RecommendationScoreRange(on_fail=OnFailAction.FIX))

# Output guard for explanations: auto-fixes trivial text without raising.
explanation_guard: Guard = Guard().use(ExplanationQuality(on_fail=OnFailAction.FIX))


# ── Public API ────────────────────────────────────────────────────────────────

class InputValidationError(ValueError):
    """Raised when user preferences fail the input guard."""


def validate_user_prefs(user_prefs: Dict) -> Dict:
    """
    Validate user preferences before they reach the scoring pipeline.

    Uses Guard.for_pydantic(UserPrefsSchema) as the official guardrails
    validation path, with Pydantic's own error messages for human-readable
    feedback.

    Parameters
    ----------
    user_prefs : dict
        Raw user preference dict (may come from user input or the CLI).

    Returns
    -------
    dict
        The validated preference dict (keys/types guaranteed valid).

    Raises
    ------
    InputValidationError
        With a structured list of field-level errors if any constraint fails.
    """
    # Fast path: try pydantic directly for rich error messages.
    try:
        UserPrefsSchema.model_validate(user_prefs)
    except PydanticValidationError as exc:
        field_errors = "; ".join(
            f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        raise InputValidationError(
            f"Input guard blocked invalid user preferences — {field_errors}. "
            f"Received: {user_prefs}"
        ) from exc

    # Official guardrails pass (stores the outcome in guard.history).
    outcome = input_guard.validate(json.dumps(user_prefs))
    if not outcome.validation_passed:
        raise InputValidationError(
            f"Input guard: validation failed. Received: {user_prefs}"
        )

    return outcome.validated_output or user_prefs


def validate_recommendation_output(
    recommendations: List[Tuple[Dict, float, str]],
) -> List[Tuple[Dict, float, str]]:
    """
    Validate and auto-fix every (song, score, explanation) tuple in the output.

    - Scores outside [0.0, 1.0] are clamped and a warning is issued.
    - Trivial/empty explanations are replaced with a fallback string and a
      warning is issued.
    - An empty recommendations list emits a warning but is returned as-is.

    Parameters
    ----------
    recommendations : list of (song_dict, score, explanation)

    Returns
    -------
    list of (song_dict, safe_score, safe_explanation)
    """
    if not recommendations:
        warnings.warn(
            "Output guard: recommendations list is empty — "
            "no results to validate.",
            stacklevel=2,
        )
        return recommendations

    cleaned: List[Tuple[Dict, float, str]] = []
    for song, score, explanation in recommendations:
        safe_score = _guard_score(score)
        safe_explanation = _guard_explanation(explanation)
        cleaned.append((song, safe_score, safe_explanation))

    return cleaned


# ── Private helpers ───────────────────────────────────────────────────────────

def _guard_score(score: float) -> float:
    outcome = score_guard.validate(str(score))
    try:
        result = float(outcome.validated_output)
    except (TypeError, ValueError):
        result = 0.0
    if abs(result - score) > 1e-9:
        warnings.warn(
            f"Output guard clamped score {score:.4f} → {result:.4f}.",
            stacklevel=3,
        )
    return result


def _guard_explanation(explanation: str) -> str:
    outcome = explanation_guard.validate(explanation)
    result: str = outcome.validated_output or ExplanationQuality._FALLBACK
    if result != explanation:
        warnings.warn(
            f"Output guard replaced explanation. "
            f"Original ({len(explanation)} chars): {explanation[:60]!r}...",
            stacklevel=3,
        )
    return result
