"""
VoxShield AI - Scam Intent Classification
Uses DistilBERT embeddings + trained classifier for scam detection.
"""
import os
import json
import pickle
import numpy as np
from typing import Dict, Tuple

# Global instances
_tokenizer = None
_bert_model = None
_scam_classifier = None


def load_bert_model(model_name: str = "distilbert-base-uncased"):
    """
    Load DistilBERT tokenizer and model globally.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Tuple of (tokenizer, model)
    """
    global _tokenizer, _bert_model
    if _tokenizer is None or _bert_model is None:
        from transformers import DistilBertTokenizer, DistilBertModel
        _tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        _bert_model = DistilBertModel.from_pretrained(model_name)
        _bert_model.eval()
    return _tokenizer, _bert_model


def get_text_embedding(text: str) -> np.ndarray:
    """
    Get CLS token embedding from DistilBERT for a text input.

    Args:
        text: Input text

    Returns:
        768-dim embedding vector
    """
    import torch

    tokenizer, model = load_bert_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Use CLS token representation
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding


def load_or_train_scam_classifier(model_path: str, data_path: str) -> object:
    """
    Load pre-trained scam classifier or train a new one from seed data.

    Args:
        model_path: Path to saved classifier pickle
        data_path: Path to scam_templates.json for training

    Returns:
        Trained classifier
    """
    global _scam_classifier

    if _scam_classifier is not None:
        return _scam_classifier

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            _scam_classifier = pickle.load(f)
        return _scam_classifier

    # Train from seed data
    print("[VoxShield] Training scam classifier from seed data...")
    _scam_classifier = _train_scam_classifier(data_path, model_path)
    return _scam_classifier


def _train_scam_classifier(data_path: str, save_path: str) -> object:
    """
    Train a logistic regression classifier on DistilBERT embeddings.

    Args:
        data_path: Path to JSON training data
        save_path: Path to save the trained model

    Returns:
        Trained LogisticRegression classifier
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import json

    with open(data_path, "r") as f:
        samples = json.load(f)

    print(f"[VoxShield] Extracting embeddings for {len(samples)} samples...")
    X = []
    y = []
    for sample in samples:
        emb = get_text_embedding(sample["text"])
        X.append(emb)
        y.append(sample["label"])

    X = np.array(X)
    y = np.array(y)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial"))
    ])
    clf.fit(X, y)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"[VoxShield] Scam classifier saved to {save_path}")
    return clf


def classify_scam(text: str, model_path: str, data_path: str) -> Dict:
    """
    Classify text for scam intent.

    Args:
        text: Transcript text to analyze
        model_path: Path to saved classifier
        data_path: Path to training data (for fallback training)

    Returns:
        Dict with keys: scam_probability, predicted_label, class_probabilities
    """
    if not text or text == "[No speech detected]":
        return {
            "scam_probability": 0.0,
            "predicted_label": "Normal",
            "class_probabilities": {"Normal": 1.0, "Scam": 0.0, "High Risk": 0.0}
        }

    try:
        classifier = load_or_train_scam_classifier(model_path, data_path)
        embedding = get_text_embedding(text)
        embedding = embedding.reshape(1, -1)

        probs = classifier.predict_proba(embedding)[0]
        pred_idx = int(np.argmax(probs))

        labels = ["Normal", "Scam", "High Risk"]
        predicted_label = labels[pred_idx]

        # Scam probability = P(Scam) + P(High Risk)
        scam_prob = float(probs[1] + probs[2])

        return {
            "scam_probability": round(scam_prob, 4),
            "predicted_label": predicted_label,
            "class_probabilities": {
                "Normal": round(float(probs[0]), 4),
                "Scam": round(float(probs[1]), 4),
                "High Risk": round(float(probs[2]), 4)
            }
        }

    except Exception as e:
        raise RuntimeError(f"Scam classification failed: {str(e)}")
