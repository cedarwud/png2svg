from __future__ import annotations

import re
from typing import Iterable


LEXICON_3GPP = {
    "a3": "A3",
    "a4": "A4",
    "a5": "A5",
    "ttt": "TTT",
    "hys": "Hys",
    "trigger": "trigger",
    "triggered": "triggered",
    "threshold": "Threshold",
    "th": "Th",
    "serving": "Serving",
    "neighbor": "Neighbor",
    "target": "Target",
    "beam": "beam",
    "hysteresis": "Hysteresis",
    "measurement": "Measurement",
    "event": "Event",
    "events": "Events",
    "triggers": "Triggers",
    "offset": "offset",
    "rlf": "RLF",
    "ping": "ping",
    "pong": "pong",
    "uho": "UHO",
    "ho": "HO",
    "measured": "Measured",
    "signal": "signal",
    "level": "level",
    "t": "t",
    "report": "Report",
    "sent": "sent",
    "condition": "Condition",
    "first": "first",
    "met": "met",
    "time": "Time",
    "rsrp": "RSRP",
    "rsrq": "RSRQ",
    "sinr": "SINR",
    "t_start": "t_start",
    "t_trigger": "t_trigger",
    "th_a4": "Th_A4",
    "th1_a5": "Th1_A5",
    "th2_a5": "Th2_A5",
}

LEXICON_PROJECT_ARCH = {
    "project": "Project",
    "architecture": "Architecture",
    "work": "Work",
    "package": "Package",
    "packages": "Packages",
    "platform": "Platform",
    "simulator": "Simulator",
    "metrics": "Metrics",
    "pipeline": "Pipeline",
    "integration": "Integration",
    "wp1": "WP1",
    "wp2": "WP2",
    "wp3": "WP3",
    "wp4": "WP4",
    "panel": "Panel",
}

CANONICAL_TEXT = {
    "t_3gpp_events_3panel": {
        "title": "3GPP Measurement Event Triggers (A3/A4/A5): Time-to-Trigger (TTT) and Hysteresis",
        "curve_label_serving": "Serving beam",
        "curve_label_neighbor": "Neighbor/Target beam",
        "axis_label_x": "Time (t)",
        "axis_label_y": "Measured signal level (e.g., RSRP/RSRQ/SINR)",
    }
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^A-Za-z0-9_]+")


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        curr = [i]
        for j, ch_b in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ch_a == ch_b else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def _apply_case(token: str, canonical: str) -> str:
    if token.isupper():
        return canonical.upper()
    if token.islower():
        return canonical.lower()
    if token[:1].isupper() and token[1:].islower():
        return canonical[:1].upper() + canonical[1:]
    return canonical


def _best_match(token: str, lexicon: dict[str, str], max_distance: int) -> tuple[str, int] | None:
    best = None
    token_lower = token.lower()
    for key, canonical in lexicon.items():
        dist = _levenshtein(token_lower, key)
        if dist > max_distance:
            continue
        if best is None or dist < best[1]:
            best = (canonical, dist)
            if dist == 0:
                break
    return best


def normalize_text(
    text: str,
    lexicon: dict[str, str],
    conf: float | None = None,
    min_conf: float = 0.7,
    max_distance: int = 2,
) -> tuple[str, int]:
    if not text:
        return text, 0
    tokens = _TOKEN_RE.findall(text)
    corrected = 0
    updated: list[str] = []
    for token in tokens:
        if not token or not token[0].isalnum():
            updated.append(token)
            continue
        token_lower = token.lower()
        canonical = lexicon.get(token_lower)
        if canonical is not None:
            new_token = _apply_case(token, canonical)
            if new_token != token:
                corrected += 1
            updated.append(new_token)
            continue
        match = _best_match(token, lexicon, max_distance)
        if match is None:
            updated.append(token)
            continue
        candidate, dist = match
        if conf is None or conf < min_conf or dist <= max_distance:
            new_token = _apply_case(token, candidate)
            if new_token != token:
                corrected += 1
            updated.append(new_token)
        else:
            updated.append(token)
    return "".join(updated), corrected


def normalize_text_items(
    items: list[dict[str, object]],
    lexicon: dict[str, str],
    min_conf: float = 0.7,
    max_distance: int = 2,
) -> int:
    total = 0
    for item in items:
        text = str(item.get("content") or item.get("text") or "")
        if not text:
            continue
        try:
            conf = float(item.get("conf", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        normalized, corrected = normalize_text(
            text, lexicon, conf=conf, min_conf=min_conf, max_distance=max_distance
        )
        if corrected:
            item["content"] = normalized
            item["text"] = normalized
        total += corrected
    return total


def text_sanity(text: str, min_alnum_ratio: float = 0.35) -> bool:
    value = text.strip()
    if not value:
        return False
    total = len(value)
    alnum = sum(1 for ch in value if ch.isalnum())
    return total > 0 and (alnum / total) >= min_alnum_ratio


def lexicon_for_template(template_id: str) -> dict[str, str]:
    if template_id == "t_3gpp_events_3panel":
        return dict(LEXICON_3GPP)
    if template_id == "t_project_architecture_v1":
        return dict(LEXICON_PROJECT_ARCH)
    return {}


def canonical_text_for_template(template_id: str) -> dict[str, str]:
    return dict(CANONICAL_TEXT.get(template_id, {}))


def normalize_lines(
    lines: Iterable[str],
    lexicon: dict[str, str],
    conf: float,
    min_conf: float,
    max_distance: int = 2,
) -> tuple[list[str], int]:
    corrected_total = 0
    out: list[str] = []
    for line in lines:
        normalized, corrected = normalize_text(
            line, lexicon, conf=conf, min_conf=min_conf, max_distance=max_distance
        )
        corrected_total += corrected
        if normalized.strip():
            out.append(normalized.strip())
    return out, corrected_total
