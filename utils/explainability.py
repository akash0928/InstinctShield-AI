"""
VoxShield AI - Explainability Utilities
Generates human-readable explanations for risk assessments.
"""
from typing import Dict, List


def generate_risk_explanation(report: Dict) -> str:
    """
    Generate a human-readable explanation of the risk assessment.

    Args:
        report: Complete risk report from risk_engine.build_risk_report

    Returns:
        Explanation string
    """
    level = report["summary"]["risk_level"]
    score = report["summary"]["risk_score"]
    scam_prob = report["scam_analysis"]["scam_probability"]
    deepfake_prob = report["voice_analysis"]["deepfake_probability"]
    manip_score = report["manipulation_analysis"]["manipulation_score"]
    patterns = report["manipulation_analysis"]["detected_patterns"]

    lines = []

    if level == "SAFE":
        lines.append(f"✅ This call appears **safe** (Risk Score: {score:.0f}/100).")
        lines.append("No significant indicators of scam activity, deepfake voice, or manipulation were detected.")
    elif level == "SUSPICIOUS":
        lines.append(f"⚠️ This call shows **suspicious patterns** (Risk Score: {score:.0f}/100).")
        if scam_prob > 0.3:
            lines.append(f"• Scam intent detected with {scam_prob*100:.0f}% probability.")
        if deepfake_prob > 0.3:
            lines.append(f"• Voice may be artificially generated ({deepfake_prob*100:.0f}% synthetic probability).")
        if patterns:
            lines.append(f"• Manipulation tactics detected: {', '.join([p.replace('_', ' ') for p in patterns])}.")
    else:  # HIGH RISK
        lines.append(f"🚨 This call is **HIGH RISK** (Risk Score: {score:.0f}/100). Do NOT comply with requests.")
        if scam_prob > 0.5:
            lines.append(f"• Strong scam indicators detected ({scam_prob*100:.0f}% probability).")
        if deepfake_prob > 0.5:
            lines.append(f"• Voice is likely artificially synthesized ({deepfake_prob*100:.0f}% synthetic).")
        if patterns:
            pattern_names = [_flag_to_readable(p) for p in patterns]
            lines.append(f"• Dangerous manipulation patterns found: {', '.join(pattern_names)}.")

    return "\n".join(lines)


def _flag_to_readable(flag: str) -> str:
    """Convert flag name to readable label."""
    mapping = {
        "otp_extraction": "OTP/Code Extraction",
        "authority_impersonation": "Authority Impersonation",
        "urgency_pressure": "Urgency Pressure",
        "isolation_tactics": "Isolation Tactics",
        "financial_pressure": "Financial Pressure"
    }
    return mapping.get(flag, flag.replace("_", " ").title())


def highlight_flagged_sentences(transcript: str, flagged_sentences: List[Dict]) -> str:
    """
    Create a version of the transcript with flagged sentences marked.

    Args:
        transcript: Full transcript text
        flagged_sentences: List of {sentence, flags} dicts

    Returns:
        Annotated transcript with markers
    """
    if not flagged_sentences:
        return transcript

    annotated = transcript
    for item in flagged_sentences:
        sentence = item["sentence"]
        flags = item["flags"]
        if sentence in annotated:
            flag_labels = " | ".join([_flag_to_readable(f) for f in flags])
            replacement = f"**[⚠️ {flag_labels}]** {sentence}"
            annotated = annotated.replace(sentence, replacement, 1)

    return annotated


def get_recommendations(report: Dict) -> List[str]:
    """
    Generate actionable recommendations based on risk analysis.

    Args:
        report: Complete risk report

    Returns:
        List of recommendation strings
    """
    level = report["summary"]["risk_level"]
    patterns = report["manipulation_analysis"]["detected_patterns"]
    recommendations = []

    if level == "SAFE":
        recommendations.append("Continue the conversation normally.")
        recommendations.append("Remain vigilant for any future suspicious requests.")
    else:
        recommendations.append("Do NOT share personal information, OTPs, or bank details.")
        recommendations.append("Hang up and call back the official number independently.")

        if "authority_impersonation" in patterns:
            recommendations.append("Verify officer/agency identity through official government websites.")

        if "otp_extraction" in patterns:
            recommendations.append("NEVER share OTPs or verification codes — legitimate organizations never ask for them.")

        if "isolation_tactics" in patterns:
            recommendations.append("Talk to a trusted family member or friend before taking any action.")

        if "financial_pressure" in patterns:
            recommendations.append("Do not transfer money or buy gift cards under any pressure.")

        if level == "HIGH RISK":
            recommendations.append("Report this number to your national cybercrime portal immediately.")
            recommendations.append("Block the caller and warn others in your contact list.")

    return recommendations
