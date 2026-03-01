"""
VoxShield AI - Deepfake Voice Detection
Uses Wav2Vec2 embeddings + LogisticRegression classifier.
When no trained model exists, uses acoustic heuristics for zero-shot detection.
"""
import os
import pickle
import numpy as np
from typing import Dict


# Global classifier instance
_deepfake_classifier = None


def load_deepfake_classifier(model_path: str):
    """
    Load pre-trained deepfake classifier if it exists.

    Args:
        model_path: Path to saved classifier pickle

    Returns:
        Classifier or None if not found
    """
    global _deepfake_classifier
    if _deepfake_classifier is not None:
        return _deepfake_classifier

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            _deepfake_classifier = pickle.load(f)
        return _deepfake_classifier

    return None


def _heuristic_deepfake_score(audio: np.ndarray, sr: int) -> float:
    """
    Heuristic-based deepfake scoring using acoustic analysis.
    Used as fallback when no trained classifier is available.

    Synthetic voices often have:
    - Unusually consistent pitch
    - Lower spectral variance
    - Artifacts in specific frequency bands
    - Unnaturally regular rhythm

    Args:
        audio: Audio array
        sr: Sample rate

    Returns:
        Synthetic probability estimate (0-1)
    """
    import librosa

    scores = []

    # 1. Pitch regularity (natural voices have more variation)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        voiced_f0 = f0[voiced_flag == 1] if voiced_flag is not None else np.array([])
        if len(voiced_f0) > 10:
            pitch_cv = np.std(voiced_f0) / (np.mean(voiced_f0) + 1e-6)
            # Low pitch variation = more synthetic
            natural_cv_threshold = 0.15
            pitch_score = max(0, 1.0 - (pitch_cv / natural_cv_threshold))
            scores.append(pitch_score * 0.4)
    except Exception:
        pass

    # 2. Spectral flatness (synthetic often has flatter spectrum)
    try:
        spec_flat = librosa.feature.spectral_flatness(y=audio)
        mean_flatness = float(np.mean(spec_flat))
        # Higher flatness = more noise-like/synthetic
        flat_score = min(mean_flatness * 15.0, 1.0)
        scores.append(flat_score * 0.2)
    except Exception:
        pass

    # 3. MFCC delta variance (natural speech has more dynamic change)
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfccs)
        delta_var = float(np.mean(np.var(mfcc_delta, axis=1)))
        # Low delta variance = less natural dynamics = more synthetic
        natural_delta_threshold = 5.0
        delta_score = max(0, 1.0 - (delta_var / natural_delta_threshold))
        scores.append(delta_score * 0.25)
    except Exception:
        pass

    # 4. Formant regularity via LPC
    try:
        from scipy.signal import lfilter
        # Simple regularity check via ZCR consistency
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_cv = float(np.std(zcr) / (np.mean(zcr) + 1e-6))
        zcr_score = max(0, 1.0 - zcr_cv * 2.0)
        scores.append(zcr_score * 0.15)
    except Exception:
        pass

    if not scores:
        return 0.3  # Default moderate uncertainty

    return round(float(np.clip(sum(scores), 0.0, 1.0)), 4)


def detect_deepfake(audio: np.ndarray, sr: int, model_path: str, use_wav2vec: bool = True) -> Dict:
    """
    Detect whether the audio contains a synthetic/deepfake voice.

    Args:
        audio: Preprocessed audio array
        sr: Sample rate
        model_path: Path to saved deepfake classifier
        use_wav2vec: Whether to use Wav2Vec2 features (recommended)

    Returns:
        Dict with keys:
            - deepfake_probability: float 0-1
            - is_synthetic: bool
            - method: str indicating detection method used
    """
    classifier = load_deepfake_classifier(model_path)

    if classifier is not None:
        # Use trained classifier
        try:
            from utils.voice_features import get_full_feature_vector
            features = get_full_feature_vector(audio, sr, use_wav2vec=use_wav2vec)
            features = features.reshape(1, -1)
            probs = classifier.predict_proba(features)[0]
            # Assume class 1 = synthetic
            synthetic_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            return {
                "deepfake_probability": round(synthetic_prob, 4),
                "is_synthetic": synthetic_prob > 0.5,
                "method": "trained_classifier"
            }
        except Exception as e:
            pass

    # Fallback: heuristic analysis
    synthetic_prob = _heuristic_deepfake_score(audio, sr)
    return {
        "deepfake_probability": synthetic_prob,
        "is_synthetic": synthetic_prob > 0.5,
        "method": "acoustic_heuristics"
    }


def train_deepfake_classifier(real_features: np.ndarray, synthetic_features: np.ndarray, save_path: str) -> object:
    """
    Train a deepfake classifier given real and synthetic voice features.
    Call this function if you have labeled audio data.

    Args:
        real_features: Feature matrix for real voices [N, D]
        synthetic_features: Feature matrix for synthetic voices [N, D]
        save_path: Path to save trained classifier

    Returns:
        Trained classifier
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = np.vstack([real_features, synthetic_features])
    y = np.array([0] * len(real_features) + [1] * len(synthetic_features))

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0))
    ])
    clf.fit(X, y)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(clf, f)

    return clf
