"""
VoxShield AI - Risk Scoring Engine
Combines scam, deepfake, and manipulation signals into a final risk score.
"""
from typing import Dict


def compute_risk_score(
    scam_probability: float,
    deepfake_probability: float,
    manipulation_score: float,
    weight_scam: float = 0.5,
    weight_deepfake: float = 0.3,
    weight_manipulation: float = 0.2
) -> float:
    """
    Compute weighted risk score from individual signal probabilities.

    Formula:
        risk_score = 100 * (w_scam * scam_prob + w_deepfake * deepfake_prob + w_manip * manip_score)

    Args:
        scam_probability: Scam intent probability (0-1)
        deepfake_probability: Deepfake voice probability (0-1)
        manipulation_score: Psychological manipulation score (0-1)
        weight_scam: Weight for scam signal (default 0.5)
        weight_deepfake: Weight for deepfake signal (default 0.3)
        weight_manipulation: Weight for manipulation signal (default 0.2)

    Returns:
        Risk score (0-100)
    """
    score = (
        weight_scam * scam_probability +
        weight_deepfake * deepfake_probability +
        weight_manipulation * manipulation_score
    )
    return round(float(score * 100), 2)


def get_risk_level(risk_score: float) -> str:
    """
    Convert numeric risk score to categorical risk level.

    Args:
        risk_score: Risk score (0-100)

    Returns:
        Risk level string: "SAFE", "SUSPICIOUS", or "HIGH RISK"
    """
    if risk_score < 30:
        return "SAFE"
    elif risk_score < 70:
        return "SUSPICIOUS"
    else:
        return "HIGH RISK"


def get_risk_color(risk_level: str) -> str:
    """
    Get display color for risk level.

    Args:
        risk_level: Risk level string

    Returns:
        Color string for Streamlit display
    """
    colors = {
        "SAFE": "green",
        "SUSPICIOUS": "orange",
        "HIGH RISK": "red"
    }
    return colors.get(risk_level, "gray")


def build_risk_report(
    transcript: str,
    scam_result: Dict,
    deepfake_result: Dict,
    manipulation_result: Dict,
    audio_duration: float
) -> Dict:
    """
    Build a comprehensive structured risk report.

    Args:
        transcript: Transcribed text
        scam_result: Output from scam classifier
        deepfake_result: Output from deepfake detector
        manipulation_result: Output from manipulation analyzer
        audio_duration: Duration of analyzed audio in seconds

    Returns:
        Complete structured JSON-serializable risk report
    """
    risk_score = compute_risk_score(
        scam_probability=scam_result.get("scam_probability", 0.0),
        deepfake_probability=deepfake_result.get("deepfake_probability", 0.0),
        manipulation_score=manipulation_result.get("manipulation_score", 0.0)
    )
    risk_level = get_risk_level(risk_score)

    return {
        "summary": {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "audio_duration_seconds": round(audio_duration, 1)
        },
        "transcript": transcript,
        "scam_analysis": {
            "scam_probability": scam_result.get("scam_probability", 0.0),
            "predicted_label": scam_result.get("predicted_label", "Unknown"),
            "class_probabilities": scam_result.get("class_probabilities", {})
        },
        "voice_analysis": {
            "deepfake_probability": deepfake_result.get("deepfake_probability", 0.0),
            "is_synthetic": deepfake_result.get("is_synthetic", False),
            "detection_method": deepfake_result.get("method", "unknown")
        },
        "manipulation_analysis": {
            "manipulation_score": manipulation_result.get("manipulation_score", 0.0),
            "detected_patterns": manipulation_result.get("manipulation_flags", []),
            "category_counts": manipulation_result.get("category_counts", {}),
            "flagged_sentences": manipulation_result.get("flagged_sentences", [])
        }
    }
