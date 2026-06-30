"""
challenge_config.py

Shared settings for the Face the AI emotion challenge.
"""

CHALLENGE_NAME = "Face the AI: Emotion Recognition Challenge"

CHALLENGE_EMOTIONS = [
    ("happy", "Happy", "😁"),
    ("sad", "Sad", "😢"),
    ("neutral", "Neutral", "😐"),
]

SUPPORTED_LABELS = {label for label, _, _ in CHALLENGE_EMOTIONS}
EMOJI_MAP = {label: emoji for label, _, emoji in CHALLENGE_EMOTIONS}

DATA_DIR = "data"
CSV_PATH = f"{DATA_DIR}/dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/emotion_challenge_model.joblib"

BASE_ROUND_SECONDS = 7
MIN_ROUND_SECONDS = 4
LEVEL_UP_EVERY = 3
BASE_AI_CONFIDENCE = 62
CONFIDENCE_STEP = 4
MAX_AI_CONFIDENCE = 80