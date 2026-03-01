# 🛡️ VoxShield AI

**Fully local, open-source AI system for detecting scam calls and deepfake voice fraud.**

---

## Features

| Capability | Technology |
|---|---|
| Speech-to-Text | `faster-whisper` (base, int8 CPU) |
| Scam Intent Detection | `DistilBERT` + LogisticRegression |
| Deepfake Voice Detection | Acoustic heuristics + optional `Wav2Vec2` |
| Manipulation Pattern Detection | Regex + semantic rule engine |
| Risk Scoring | Weighted aggregation formula |
| UI | Streamlit |

**No paid APIs. No cloud. Runs entirely on CPU.**

---

## Installation

```bash
# 1. Clone or unzip the project
cd voxshield

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** First run downloads ~300MB of model files (Whisper base, DistilBERT) from HuggingFace. After that, everything is cached locally.

---

## Running

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Usage

### Upload Mode
1. Click **Upload Mode** (default)
2. Upload an `.mp3` or `.wav` file (max 60 seconds)
3. Click **Analyze Audio**
4. View results: risk score, transcript, flagged patterns, recommendations

### Demo Mode
- Select a preset from the **Demo Samples** dropdown
- Click **Analyze Audio** to see the system in action without needing audio files

---

## Project Structure

```
voxshield/
├── app.py                    # Streamlit UI
├── pipeline.py               # Main analysis orchestrator
├── config.py                 # Central configuration
│
├── models/
│   ├── scam_model.pkl        # Auto-trained on first run
│   └── deepfake_model.pkl    # Optional - place trained model here
│
├── utils/
│   ├── audio_preprocess.py   # Load, trim, normalize audio
│   ├── stt.py                # Whisper speech-to-text
│   ├── voice_features.py     # Wav2Vec2 + acoustic features
│   ├── scam_inference.py     # DistilBERT scam classifier
│   ├── deepfake_detection.py # Voice authenticity detection
│   ├── manipulation.py       # Pattern detection engine
│   ├── explainability.py     # Human-readable explanations
│   └── risk_engine.py        # Risk scoring formula
│
├── data/
│   └── scam_templates.json   # Seed training data for scam classifier
│
├── requirements.txt
└── README.md
```

---

## Risk Score Formula

```
risk_score = 100 × (0.5 × scam_probability + 0.3 × deepfake_probability + 0.2 × manipulation_score)

0–29  → ✅ SAFE
30–69 → ⚠️ SUSPICIOUS
70+   → 🚨 HIGH RISK
```

---

## Detected Manipulation Patterns

| Pattern | Description |
|---|---|
| 🔐 OTP Extraction | Attempts to collect verification codes |
| 👮 Authority Impersonation | Claiming to be police, CBI, IRS, bank officials |
| ⏰ Urgency Pressure | "Act now", "30 minutes", deadline language |
| 🔇 Isolation Tactics | "Don't tell anyone", "stay on the line" |
| 💰 Financial Pressure | Wire transfers, gift cards, crypto, frozen accounts |

---

## Training Your Own Deepfake Detector

If you have labeled real vs. synthetic audio samples:

```python
from utils.voice_features import get_full_feature_vector
from utils.deepfake_detection import train_deepfake_classifier
import numpy as np

# Extract features for each file
real_features = np.array([get_full_feature_vector(audio, sr) for audio, sr in real_samples])
synth_features = np.array([get_full_feature_vector(audio, sr) for audio, sr in synth_samples])

train_deepfake_classifier(real_features, synth_features, "models/deepfake_model.pkl")
```

---

## Performance

| Stage | Typical Time (CPU) |
|---|---|
| Audio Preprocessing | ~0.1s |
| Whisper Transcription (30s audio) | 2–5s |
| Scam Classification | 0.5–1s |
| Voice Feature Extraction | 1–3s |
| Manipulation Detection | <0.1s |
| **Total** | **~5–10s** |

---

## License

MIT License — free to use, modify, and distribute.
