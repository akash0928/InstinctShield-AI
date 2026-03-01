"""
VoxShield AI - Main Analysis Pipeline
Orchestrates all components for end-to-end scam call detection.
"""
import os
import sys
import time
import numpy as np
from typing import Dict, Optional

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.audio_preprocess import preprocess_audio, save_temp_wav
from utils.stt import transcribe_audio
from utils.scam_inference import classify_scam
from utils.deepfake_detection import detect_deepfake
from utils.manipulation import analyze_manipulation
from utils.risk_engine import build_risk_report
from utils.explainability import generate_risk_explanation, get_recommendations, highlight_flagged_sentences


def analyze_audio_file(
    file_path: str,
    progress_callback=None
) -> Dict:
    """
    Run full VoxShield analysis pipeline on an audio file.

    Args:
        file_path: Path to .mp3 or .wav audio file
        progress_callback: Optional callable(step: str, progress: float)

    Returns:
        Complete analysis report dict with all findings
    """
    timings = {}
    start_total = time.time()

    def _update(step, progress):
        if progress_callback:
            progress_callback(step, progress)

    # ─── Step 1: Audio Preprocessing ───────────────────────────────────────
    _update("Preprocessing audio...", 0.05)
    t0 = time.time()
    audio, sr, duration = preprocess_audio(
        file_path,
        max_seconds=config.MAX_AUDIO_DURATION_SECONDS,
        target_sr=config.AUDIO_SAMPLE_RATE
    )
    timings["preprocess"] = round(time.time() - t0, 2)

    # ─── Step 2: Speech-to-Text ─────────────────────────────────────────────
    _update("Transcribing audio...", 0.20)
    t0 = time.time()
    transcript = transcribe_audio(audio, sr, model_size=config.WHISPER_MODEL_SIZE)
    timings["stt"] = round(time.time() - t0, 2)

    # ─── Step 3: Scam Intent Classification ────────────────────────────────
    _update("Analyzing scam intent...", 0.45)
    t0 = time.time()
    scam_result = classify_scam(
        transcript,
        model_path=config.SCAM_MODEL_PATH,
        data_path=os.path.join(config.DATA_DIR, "scam_templates.json")
    )
    timings["scam"] = round(time.time() - t0, 2)

    # ─── Step 4: Deepfake Voice Detection ──────────────────────────────────
    _update("Detecting voice authenticity...", 0.65)
    t0 = time.time()
    deepfake_result = detect_deepfake(
        audio, sr,
        model_path=config.DEEPFAKE_MODEL_PATH,
        use_wav2vec=True
    )
    timings["deepfake"] = round(time.time() - t0, 2)

    # ─── Step 5: Manipulation Pattern Detection ─────────────────────────────
    _update("Detecting manipulation patterns...", 0.80)
    t0 = time.time()
    manipulation_result = analyze_manipulation(transcript, config.MANIPULATION_PATTERNS)
    timings["manipulation"] = round(time.time() - t0, 2)

    # ─── Step 6: Risk Scoring ───────────────────────────────────────────────
    _update("Computing risk score...", 0.90)
    report = build_risk_report(
        transcript=transcript,
        scam_result=scam_result,
        deepfake_result=deepfake_result,
        manipulation_result=manipulation_result,
        audio_duration=duration
    )

    # ─── Step 7: Explainability ─────────────────────────────────────────────
    report["explanation"] = generate_risk_explanation(report)
    report["recommendations"] = get_recommendations(report)
    report["highlighted_transcript"] = highlight_flagged_sentences(
        transcript,
        manipulation_result.get("flagged_sentences", [])
    )
    report["timings"] = timings
    report["total_time"] = round(time.time() - start_total, 2)

    _update("Analysis complete.", 1.0)
    return report


def warm_up_models():
    """
    Pre-load all models at startup to avoid cold-start delays during analysis.
    Call this once when the app initializes.
    """
    print("[VoxShield] Warming up models...")

    # Load Whisper
    from utils.stt import load_whisper_model
    load_whisper_model(
        model_size=config.WHISPER_MODEL_SIZE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
        device=config.WHISPER_DEVICE
    )
    print("[VoxShield] ✓ Whisper loaded")

    # Load DistilBERT
    from utils.scam_inference import load_bert_model
    load_bert_model(config.SCAM_MODEL_NAME)
    print("[VoxShield] ✓ DistilBERT loaded")

    # Load or train scam classifier
    from utils.scam_inference import load_or_train_scam_classifier
    load_or_train_scam_classifier(
        config.SCAM_MODEL_PATH,
        os.path.join(config.DATA_DIR, "scam_templates.json")
    )
    print("[VoxShield] ✓ Scam classifier ready")

    # Optionally pre-load Wav2Vec2
    try:
        from utils.voice_features import load_wav2vec_model
        load_wav2vec_model(config.WAV2VEC_MODEL_NAME)
        print("[VoxShield] ✓ Wav2Vec2 loaded")
    except Exception as e:
        print(f"[VoxShield] ⚠ Wav2Vec2 skipped: {e}")

    print("[VoxShield] All models ready.")
