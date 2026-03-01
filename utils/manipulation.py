"""
VoxShield AI - Psychological Manipulation Pattern Detection
Detects OTP extraction, authority impersonation, urgency pressure, and isolation tactics.
"""
import re
import json
import numpy as np
from typing import Dict, List, Tuple


def _load_patterns(config_patterns: Dict) -> Dict:
    """
    Compile regex patterns from config.

    Args:
        config_patterns: Dict of category -> list of pattern strings

    Returns:
        Dict of category -> list of compiled regex patterns
    """
    compiled = {}
    for category, patterns in config_patterns.items():
        compiled[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
    return compiled


def detect_manipulation_in_sentence(sentence: str, compiled_patterns: Dict) -> List[str]:
    """
    Detect which manipulation categories are present in a sentence.

    Args:
        sentence: Input sentence
        compiled_patterns: Dict of compiled regex patterns

    Returns:
        List of matching category names
    """
    flags = []
    for category, patterns in compiled_patterns.items():
        for pattern in patterns:
            if pattern.search(sentence):
                flags.append(category)
                break  # One match per category per sentence
    return flags


def split_into_sentences(text: str) -> List[str]:
    """
    Split transcript into sentences for granular analysis.

    Args:
        text: Input transcript

    Returns:
        List of sentences
    """
    # Split on sentence-ending punctuation or long pauses indicated by ellipsis
    sentences = re.split(r'(?<=[.!?])\s+|(?<=\.{3})\s*', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    return sentences if sentences else [text]


def compute_manipulation_score(
    flagged_sentences: List[Dict],
    total_sentences: int,
    category_weights: Dict = None
) -> float:
    """
    Compute a normalized manipulation score (0-1).

    Args:
        flagged_sentences: List of dicts with 'sentence' and 'flags'
        total_sentences: Total number of sentences analyzed
        category_weights: Optional weights per category (default equal)

    Returns:
        Manipulation score between 0 and 1
    """
    if total_sentences == 0:
        return 0.0

    default_weights = {
        "otp_extraction": 1.0,
        "authority_impersonation": 0.9,
        "urgency_pressure": 0.8,
        "isolation_tactics": 0.95,
        "financial_pressure": 1.0
    }
    weights = category_weights or default_weights

    total_weight = 0.0
    for item in flagged_sentences:
        for flag in item["flags"]:
            total_weight += weights.get(flag, 0.7)

    # Normalize: cap at 5x average weight to get 1.0
    max_possible = total_sentences * max(default_weights.values())
    score = min(total_weight / max(max_possible, 1.0), 1.0)

    # Boost if high-severity flags present
    high_severity_categories = {"otp_extraction", "isolation_tactics", "financial_pressure"}
    all_flags = {flag for item in flagged_sentences for flag in item["flags"]}
    if high_severity_categories & all_flags:
        score = min(score * 1.3, 1.0)

    return round(float(score), 4)


def analyze_manipulation(text: str, patterns_config: Dict) -> Dict:
    """
    Full manipulation analysis pipeline.

    Args:
        text: Transcript text
        patterns_config: Config dict with manipulation patterns

    Returns:
        Dict with keys:
            - manipulation_flags: list of all unique flagged categories
            - flagged_sentences: list of {sentence, flags} dicts
            - manipulation_score: float 0-1
            - category_counts: dict of category -> count
    """
    if not text or text == "[No speech detected]":
        return {
            "manipulation_flags": [],
            "flagged_sentences": [],
            "manipulation_score": 0.0,
            "category_counts": {}
        }

    compiled_patterns = _load_patterns(patterns_config)
    sentences = split_into_sentences(text)

    flagged_sentences = []
    all_flags = set()
    category_counts = {}

    for sentence in sentences:
        flags = detect_manipulation_in_sentence(sentence, compiled_patterns)
        if flags:
            flagged_sentences.append({
                "sentence": sentence,
                "flags": flags
            })
            for flag in flags:
                all_flags.add(flag)
                category_counts[flag] = category_counts.get(flag, 0) + 1

    manipulation_score = compute_manipulation_score(flagged_sentences, len(sentences))

    return {
        "manipulation_flags": sorted(list(all_flags)),
        "flagged_sentences": flagged_sentences,
        "manipulation_score": manipulation_score,
        "category_counts": category_counts
    }


def format_flag_name(flag: str) -> str:
    """
    Convert snake_case flag name to human-readable format.

    Args:
        flag: Snake case flag name

    Returns:
        Human-readable string
    """
    mapping = {
        "otp_extraction": "🔐 OTP/Code Extraction",
        "authority_impersonation": "👮 Authority Impersonation",
        "urgency_pressure": "⏰ Urgency Pressure",
        "isolation_tactics": "🔇 Isolation Tactics",
        "financial_pressure": "💰 Financial Pressure"
    }
    return mapping.get(flag, flag.replace("_", " ").title())
