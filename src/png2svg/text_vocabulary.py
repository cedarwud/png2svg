"""Text vocabulary management for OCR noise filtering."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_VOCABULARY_DIR = Path(__file__).resolve().parents[2] / "config" / "vocabulary"
_CACHE: dict[str, dict[str, Any]] = {}


def load_vocabulary(template_id: str) -> dict[str, Any] | None:
    """Load vocabulary configuration for a template.

    Args:
        template_id: Template identifier (e.g., 't_3gpp_events_3panel')

    Returns:
        Vocabulary configuration dict or None if not found
    """
    if template_id in _CACHE:
        return _CACHE[template_id]

    vocab_path = _VOCABULARY_DIR / f"{template_id}.yaml"
    if not vocab_path.exists():
        return None

    try:
        with open(vocab_path) as f:
            vocab = yaml.safe_load(f)
        _CACHE[template_id] = vocab
        return vocab
    except Exception:
        return None


def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    """Compile regex patterns, ignoring invalid ones."""
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            continue
    return compiled


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _similarity_ratio(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings (0.0 to 1.0)."""
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    distance = _levenshtein_distance(s1.lower(), s2.lower())
    return 1.0 - (distance / max_len)


def is_noise_text(text: str, vocab: dict[str, Any] | None) -> bool:
    """Check if text matches known noise patterns.

    Args:
        text: Text to check
        vocab: Vocabulary configuration

    Returns:
        True if text is likely OCR noise
    """
    if not vocab:
        return False

    noise_patterns = vocab.get("noise_patterns", [])
    if not noise_patterns:
        return False

    compiled = _compile_patterns(noise_patterns)
    text_clean = text.strip()

    for pattern in compiled:
        if pattern.search(text_clean):
            return True

    return False


def matches_valid_pattern(text: str, vocab: dict[str, Any] | None) -> bool:
    """Check if text matches any valid pattern.

    Args:
        text: Text to check
        vocab: Vocabulary configuration

    Returns:
        True if text matches a valid pattern
    """
    if not vocab:
        return True  # No vocab means no filtering

    valid_patterns = vocab.get("valid_patterns", [])
    if not valid_patterns:
        return True

    compiled = _compile_patterns(valid_patterns)
    text_clean = text.strip()

    for pattern in compiled:
        if pattern.match(text_clean):
            return True

    return False


def fuzzy_match_vocabulary(
    text: str,
    vocab: dict[str, Any] | None,
    min_similarity: float = 0.75,
) -> tuple[bool, float, str | None]:
    """Check if text fuzzy-matches any vocabulary entry.

    Args:
        text: Text to check
        vocab: Vocabulary configuration
        min_similarity: Minimum similarity threshold

    Returns:
        Tuple of (matches, best_similarity, matched_term)
    """
    if not vocab:
        return (True, 1.0, None)

    fuzzy_cfg = vocab.get("fuzzy_match", {})
    if not fuzzy_cfg.get("enabled", True):
        return (True, 1.0, None)

    min_sim = fuzzy_cfg.get("min_similarity", min_similarity)
    max_edit = fuzzy_cfg.get("max_edit_distance", 2)

    # Collect all vocabulary terms
    terms: list[str] = []
    terms.extend(vocab.get("required_exact", []))
    terms.extend(vocab.get("required_partial", []))

    if not terms:
        return (True, 1.0, None)

    text_lower = text.lower().strip()
    best_sim = 0.0
    best_term = None

    for term in terms:
        term_lower = term.lower()

        # Exact match
        if text_lower == term_lower:
            return (True, 1.0, term)

        # Substring match (text contains term or term contains text)
        if term_lower in text_lower or text_lower in term_lower:
            return (True, 1.0, term)

        # Fuzzy match
        sim = _similarity_ratio(text_lower, term_lower)
        if sim > best_sim:
            best_sim = sim
            best_term = term

        # Also check individual words
        for word in text_lower.split():
            if len(word) < 3:
                continue
            word_sim = _similarity_ratio(word, term_lower)
            if word_sim > best_sim:
                best_sim = word_sim
                best_term = term

    # Check edit distance for short terms
    if best_term and len(text_lower) <= 10:
        edit_dist = _levenshtein_distance(text_lower, best_term.lower())
        if edit_dist <= max_edit:
            return (True, best_sim, best_term)

    return (best_sim >= min_sim, best_sim, best_term)


def get_confidence_threshold(role: str, vocab: dict[str, Any] | None) -> float:
    """Get confidence threshold for a text role.

    Args:
        role: Text role (e.g., 'title', 'annotation')
        vocab: Vocabulary configuration

    Returns:
        Confidence threshold (0.0 to 1.0)
    """
    if not vocab:
        return 0.5

    thresholds = vocab.get("confidence_thresholds", {})
    return float(thresholds.get(role, thresholds.get("default", 0.5)))


def validate_text_item(
    item: dict[str, Any],
    vocab: dict[str, Any] | None,
    strict: bool = False,
) -> tuple[bool, str]:
    """Validate a text item against vocabulary.

    Args:
        item: Text item dict with 'text', 'conf', 'role' fields
        vocab: Vocabulary configuration
        strict: If True, require positive pattern match

    Returns:
        Tuple of (is_valid, reason)
    """
    text = str(item.get("content") or item.get("text") or "").strip()
    if not text:
        return (False, "empty_text")

    conf = float(item.get("conf", item.get("confidence", 0.5)))
    role = str(item.get("role", "annotation"))

    # Check noise patterns first
    if is_noise_text(text, vocab):
        return (False, "noise_pattern")

    # Check confidence threshold
    min_conf = get_confidence_threshold(role, vocab)
    if conf < min_conf:
        # Allow if matches valid pattern
        if matches_valid_pattern(text, vocab):
            pass  # Override low confidence
        else:
            return (False, f"low_confidence_{conf:.2f}<{min_conf:.2f}")

    # Check fuzzy match
    matches, similarity, matched_term = fuzzy_match_vocabulary(text, vocab)
    if not matches and strict:
        return (False, f"no_vocab_match_{similarity:.2f}")

    # Additional heuristics for common OCR errors
    if _has_suspicious_pattern(text):
        if not matches:
            return (False, "suspicious_pattern")

    return (True, "valid")


def _has_suspicious_pattern(text: str) -> bool:
    """Check for suspicious OCR patterns."""
    # Random character sequences
    if re.match(r"^[a-z]{2,4}\s+[a-z]{2,4}$", text, re.IGNORECASE):
        return True

    # Too many isolated single characters
    words = text.split()
    single_chars = sum(1 for w in words if len(w) == 1)
    if len(words) >= 3 and single_chars / len(words) > 0.4:
        return True

    # Repeated characters (OCR artifact)
    if re.search(r"(.)\1{3,}", text):
        return True

    return False


def deduplicate_text_items(
    items: list[dict[str, Any]],
    distance_threshold: float = 20.0,
) -> list[dict[str, Any]]:
    """Remove duplicate text items based on content and position.

    Args:
        items: List of text items
        distance_threshold: Maximum pixel distance for duplicates

    Returns:
        Deduplicated list
    """
    if not items:
        return items

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()

    for item in items:
        text = str(item.get("content") or item.get("text") or "").strip().lower()
        if not text:
            continue

        bbox = item.get("bbox", {})
        x = int(round(float(bbox.get("x", 0))))
        y = int(round(float(bbox.get("y", 0))))

        # Check for exact or near duplicates
        is_dup = False
        for seen_text, seen_x, seen_y in seen:
            if text == seen_text or _similarity_ratio(text, seen_text) > 0.85:
                dist = ((x - seen_x) ** 2 + (y - seen_y) ** 2) ** 0.5
                if dist < distance_threshold:
                    is_dup = True
                    break

        if not is_dup:
            seen.add((text, x, y))
            result.append(item)

    return result
