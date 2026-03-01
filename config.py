"""
VoxShield AI - Central Configuration
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# STT Config
WHISPER_MODEL_SIZE = "base"       # "tiny", "base", "small"
WHISPER_COMPUTE_TYPE = "int8"     # CPU-optimized
WHISPER_DEVICE = "cpu"
MAX_AUDIO_DURATION_SECONDS = 60

# Scam Classifier Config
SCAM_MODEL_NAME = "distilbert-base-uncased"
SCAM_MODEL_PATH = os.path.join(MODELS_DIR, "scam_model.pkl")
SCAM_LABELS = ["Normal", "Scam", "High Risk"]

# Deepfake Detection Config
DEEPFAKE_MODEL_PATH = os.path.join(MODELS_DIR, "deepfake_model.pkl")
WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base"
AUDIO_SAMPLE_RATE = 16000
AUDIO_MAX_LENGTH_SECONDS = 30    # For feature extraction (first 30s)

# Risk Engine Weights
WEIGHT_SCAM = 0.5
WEIGHT_DEEPFAKE = 0.3
WEIGHT_MANIPULATION = 0.2

# Risk Thresholds
RISK_SAFE_MAX = 30
RISK_SUSPICIOUS_MAX = 70

# Manipulation Patterns
MANIPULATION_PATTERNS = {
    "otp_extraction": [
        r"\botp\b", r"\bone.time.password\b", r"\bverification code\b",
        r"\bsend me the code\b", r"\bshare the code\b", r"\benter the code\b",
        r"\bpin\b.*\bsend\b", r"\bsend.*\bpin\b"
    ],
    "authority_impersonation": [
        r"\bpolice\b", r"\bcbi\b", r"\bnarcotic\b", r"\bincome tax\b",
        r"\bedi\b", r"\bgovernment\b", r"\bminister\b", r"\bofficer\b",
        r"\bfbi\b", r"\bcia\b", r"\brbi\b", r"\bbank official\b",
        r"\bcustomer care\b.*\bhead\b", r"\bwill be arrested\b",
        r"\blegal action\b", r"\bwarrant\b"
    ],
    "urgency_pressure": [
        r"\bimmediately\b", r"\bright now\b", r"\burgent\b", r"\bemergency\b",
        r"\blast chance\b", r"\bexpire[sd]?\b", r"\bwithin.*hour\b",
        r"\bwithin.*minute\b", r"\bdeadline\b", r"\bact now\b",
        r"\bdon.t wait\b", r"\btime is running\b"
    ],
    "isolation_tactics": [
        r"\bdon.t tell\b", r"\bkeep.*secret\b", r"\bdon.t inform\b",
        r"\bdon.t contact\b", r"\bstay on the line\b", r"\bdon.t hang up\b",
        r"\bdon.t discuss\b.*\banyone\b", r"\bonly.*between us\b",
        r"\bnobody.*knows\b", r"\bdon.t share.*with\b"
    ],
    "financial_pressure": [
        r"\btransfer.*amount\b", r"\bsend.*money\b", r"\bpay.*fine\b",
        r"\byour account.*freeze\b", r"\bblock.*account\b", r"\bfrozen\b",
        r"\bgift card\b", r"\bitunes\b", r"\bcrypto\b.*\bsend\b",
        r"\bwire.*transfer\b", r"\bpay.*immediately\b"
    ]
}

SCAM_SEED_TEMPLATES = [
    {"text": "Your bank account has been compromised. Please share your OTP immediately to verify your identity.", "label": 2},
    {"text": "This is officer from CBI. You are involved in a money laundering case. Do not contact anyone.", "label": 2},
    {"text": "Congratulations! You have won a lottery of 10 lakhs. Send 500 rupees processing fee now.", "label": 2},
    {"text": "Your SIM card will be blocked within 2 hours. Call us back urgently to prevent this.", "label": 1},
    {"text": "Income tax department has issued a warrant for your arrest. Pay penalty or face action.", "label": 2},
    {"text": "Hello, I am calling from Amazon. There is a suspicious order on your account. Please verify.", "label": 1},
    {"text": "Your KYC is incomplete. Share your Aadhaar number and OTP to avoid account suspension.", "label": 2},
    {"text": "We are offering you a special discount on our insurance plan. Are you interested?", "label": 0},
    {"text": "Hi, I wanted to confirm your appointment for tomorrow at 10 AM. Please let me know.", "label": 0},
    {"text": "Your package is ready for delivery. Please confirm your address for shipping.", "label": 0},
    {"text": "Don't tell anyone about this call. This is a confidential government matter.", "label": 2},
    {"text": "Transfer the money within the next 30 minutes or your account will be permanently frozen.", "label": 2},
    {"text": "This is your last chance to claim your reward. Act now before the offer expires.", "label": 1},
    {"text": "We have detected unusual activity on your credit card. Stay on the line for verification.", "label": 1},
    {"text": "Your electricity bill is overdue. Pay now or service will be disconnected in 2 hours.", "label": 1},
    {"text": "Send gift cards worth $500 to clear your outstanding taxes. This is IRS official.", "label": 2},
    {"text": "Hello, this is a reminder about your car's extended warranty expiration.", "label": 0},
    {"text": "You have been selected for a government scheme. No payment required, just verify your details.", "label": 1},
    {"text": "Narcotics bureau found a package with drugs registered in your name. Legal action will follow.", "label": 2},
    {"text": "Please send the OTP received on your phone. It is for account security verification.", "label": 2},
]
