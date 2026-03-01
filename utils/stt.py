"""
VoxShield AI - Speech-to-Text Module
Uses faster-whisper for CPU-optimized transcription.
"""
import os
import numpy as np
from typing import Optional

# Global model instance (loaded once)
_whisper_model = None


def load_whisper_model(model_size: str = "base", compute_type: str = "int8", device: str = "cpu"):
    """
    Load the Whisper model globally (once at startup).

    Args:
        model_size: Whisper model size ("tiny", "base", "small")
        compute_type: Quantization type ("int8" for CPU efficiency)
        device: Device to run on ("cpu")

    Returns:
        Loaded WhisperModel instance
    """
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
    return _whisper_model


def transcribe_audio(audio: np.ndarray, sr: int, model_size: str = "base") -> str:
    """
    Transcribe audio array to text using faster-whisper.

    Args:
        audio: Preprocessed audio array (float32, 16kHz mono)
        sr: Sample rate (should be 16000)
        model_size: Whisper model size

    Returns:
        Full transcript as a string

    Raises:
        RuntimeError: If transcription fails
    """
    try:
        model = load_whisper_model(model_size=model_size)

        # faster-whisper accepts numpy arrays directly
        segments, info = model.transcribe(
            audio,
            beam_size=3,
            language=None,       # auto-detect
            vad_filter=True,     # filter silence
            vad_parameters={"min_silence_duration_ms": 500}
        )

        transcript_parts = []
        for segment in segments:
            transcript_parts.append(segment.text.strip())

        transcript = " ".join(transcript_parts).strip()
        return transcript if transcript else "[No speech detected]"

    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")


def transcribe_file(file_path: str, model_size: str = "base") -> str:
    """
    Transcribe an audio file directly by path.

    Args:
        file_path: Path to audio file
        model_size: Whisper model size

    Returns:
        Full transcript string
    """
    try:
        model = load_whisper_model(model_size=model_size)

        segments, info = model.transcribe(
            file_path,
            beam_size=3,
            language=None,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )

        transcript_parts = [segment.text.strip() for segment in segments]
        transcript = " ".join(transcript_parts).strip()
        return transcript if transcript else "[No speech detected]"

    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")
