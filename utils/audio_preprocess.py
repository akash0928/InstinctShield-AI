"""
VoxShield AI - Audio Preprocessing Utilities
Handles loading, validation, resampling, and trimming of audio files.
"""
import os
import tempfile
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple


def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to target sample rate.

    Args:
        file_path: Path to .mp3 or .wav file
        target_sr: Target sample rate (default 16000 Hz for Whisper/Wav2Vec2)

    Returns:
        Tuple of (audio_array, sample_rate)

    Raises:
        ValueError: If file format is unsupported or loading fails
    """
    if not os.path.exists(file_path):
        raise ValueError(f"Audio file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in [".mp3", ".wav", ".flac", ".ogg", ".m4a"]:
        raise ValueError(f"Unsupported audio format: {ext}")

    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {str(e)}")


def get_audio_duration(audio: np.ndarray, sr: int) -> float:
    """
    Get duration of audio in seconds.

    Args:
        audio: Audio array
        sr: Sample rate

    Returns:
        Duration in seconds
    """
    return len(audio) / sr


def trim_audio(audio: np.ndarray, sr: int, max_seconds: int = 60) -> np.ndarray:
    """
    Trim audio to maximum duration.

    Args:
        audio: Audio array
        sr: Sample rate
        max_seconds: Maximum duration in seconds

    Returns:
        Trimmed audio array
    """
    max_samples = max_seconds * sr
    if len(audio) > max_samples:
        return audio[:max_samples]
    return audio


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio amplitude to [-1, 1].

    Args:
        audio: Audio array

    Returns:
        Normalized audio array
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def save_temp_wav(audio: np.ndarray, sr: int) -> str:
    """
    Save audio array as a temporary WAV file for processing.

    Args:
        audio: Audio array
        sr: Sample rate

    Returns:
        Path to temporary WAV file
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    return tmp.name


def preprocess_audio(file_path: str, max_seconds: int = 60, target_sr: int = 16000) -> Tuple[np.ndarray, int, float]:
    """
    Full preprocessing pipeline: load, validate, trim, normalize.

    Args:
        file_path: Path to input audio file
        max_seconds: Maximum audio duration to process
        target_sr: Target sample rate

    Returns:
        Tuple of (processed_audio, sample_rate, duration_seconds)
    """
    audio, sr = load_audio(file_path, target_sr=target_sr)
    duration = get_audio_duration(audio, sr)
    audio = trim_audio(audio, sr, max_seconds=max_seconds)
    audio = normalize_audio(audio)
    return audio, sr, min(duration, max_seconds)
