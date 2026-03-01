"""
VoxShield AI - Voice Feature Extraction
Extracts Wav2Vec2 embeddings and acoustic features for deepfake detection.
"""
import numpy as np
from typing import Optional

# Global model instances
_wav2vec_processor = None
_wav2vec_model = None


def load_wav2vec_model(model_name: str = "facebook/wav2vec2-base"):
    """
    Load Wav2Vec2 model and processor globally (once at startup).

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Tuple of (processor, model)
    """
    global _wav2vec_processor, _wav2vec_model
    if _wav2vec_processor is None or _wav2vec_model is None:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        _wav2vec_processor = Wav2Vec2Processor.from_pretrained(model_name)
        _wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        _wav2vec_model.eval()
    return _wav2vec_processor, _wav2vec_model


def extract_wav2vec_embeddings(audio: np.ndarray, sr: int, model_name: str = "facebook/wav2vec2-base") -> np.ndarray:
    """
    Extract mean-pooled Wav2Vec2 embeddings from audio.

    Args:
        audio: Audio array (float32, 16kHz)
        sr: Sample rate
        model_name: Wav2Vec2 model name

    Returns:
        1D embedding vector (768-dim for base model)
    """
    import torch

    processor, model = load_wav2vec_model(model_name)

    # Truncate for speed (first 10 seconds)
    max_samples = min(len(audio), sr * 10)
    audio_chunk = audio[:max_samples]

    inputs = processor(
        audio_chunk,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pool across time dimension
    hidden_states = outputs.last_hidden_state  # [1, T, 768]
    embedding = hidden_states.mean(dim=1).squeeze().numpy()  # [768]
    return embedding


def extract_acoustic_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract traditional acoustic features as supplement to deep embeddings.
    Includes MFCCs, spectral features, and pitch statistics.

    Args:
        audio: Audio array
        sr: Sample rate

    Returns:
        Feature vector (combined acoustic statistics)
    """
    import librosa

    features = []

    # MFCCs (13 coefficients, mean + std = 26)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfccs, axis=1).tolist())
    features.extend(np.std(mfccs, axis=1).tolist())

    # Spectral centroid (mean + std)
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append(float(np.mean(spec_centroid)))
    features.append(float(np.std(spec_centroid)))

    # Spectral rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.append(float(np.mean(spec_rolloff)))
    features.append(float(np.std(spec_rolloff)))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(float(np.mean(zcr)))
    features.append(float(np.std(zcr)))

    # RMS energy
    rms = librosa.feature.rms(y=audio)
    features.append(float(np.mean(rms)))
    features.append(float(np.std(rms)))

    return np.array(features, dtype=np.float32)


def get_full_feature_vector(audio: np.ndarray, sr: int, use_wav2vec: bool = True) -> np.ndarray:
    """
    Get the complete feature vector for deepfake classification.

    Args:
        audio: Audio array
        sr: Sample rate
        use_wav2vec: Whether to include Wav2Vec2 embeddings (slower but better)

    Returns:
        Combined feature vector
    """
    acoustic = extract_acoustic_features(audio, sr)

    if use_wav2vec:
        try:
            w2v_embedding = extract_wav2vec_embeddings(audio, sr)
            return np.concatenate([acoustic, w2v_embedding])
        except Exception:
            # Fall back to acoustic only if Wav2Vec2 fails
            return acoustic

    return acoustic
