"""
game.py

STEP 3. Run this after train_model.py.

The game. The AI shows an emotion prompt, you act it out on webcam, and the
model guesses what emotion it sees. If the AI guesses correctly, it scores a
point. If it misses by the end of the round, the player scores.

Controls:
    [Q] -> quit
"""

import os
import time
import random
import cv2
import joblib

from challenge_config import (
    BASE_AI_CONFIDENCE,
    BASE_ROUND_SECONDS,
    CHALLENGE_EMOTIONS,
    CHALLENGE_NAME,
    CONFIDENCE_STEP,
    EMOJI_MAP,
    LEVEL_UP_EVERY,
    MAX_AI_CONFIDENCE,
    MIN_ROUND_SECONDS,
    MODEL_PATH,
)
from feature_extraction import extract_features
from mediapipe_compat import create_face_landmarker, detect_face_landmarks


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run data_collector.py then train_model.py first."
        )
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["encoder"]


def main():
    clf, encoder = load_model()
    classes = list(encoder.classes_)
    round_emotions = [item for item in CHALLENGE_EMOTIONS if item[0] in classes]
    if not round_emotions:
        round_emotions = [(label, label.capitalize(), EMOJI_MAP.get(label, "")) for label in classes]

    face_landmarker = create_face_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    player_score = 0
    ai_score = 0
    round_number = 1
    level = 1
    round_target = random.choice(round_emotions)
    round_start = time.time()
    round_best_label = None
    round_best_confidence = 0.0
    live_guess = "--"
    live_confidence = 0.0

    print(f"{CHALLENGE_NAME} started. Press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        results = detect_face_landmarks(face_landmarker, frame)

        elapsed = time.time() - round_start
        round_seconds = max(MIN_ROUND_SECONDS, BASE_ROUND_SECONDS - (level - 1))
        win_threshold = min(MAX_AI_CONFIDENCE, BASE_AI_CONFIDENCE + (level - 1) * CONFIDENCE_STEP)
        time_left = max(0.0, round_seconds - elapsed)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            features = extract_features(landmarks).reshape(1, -1)
            probs = clf.predict_proba(features)[0]

            prediction_idx = int(probs.argmax())
            live_guess = classes[prediction_idx]
            live_confidence = float(probs[prediction_idx]) * 100

            if live_confidence > round_best_confidence:
                round_best_confidence = live_confidence
                round_best_label = live_guess
        else:
            live_guess = "No face"
            live_confidence = 0.0

        target_label, target_name, target_emoji = round_target
        guess_color = (0, 255, 0) if live_guess == target_label and live_confidence >= win_threshold else (0, 165, 255)

        cv2.putText(frame, f"Act it out: {target_name.upper()} {target_emoji}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"AI guess: {str(live_guess).upper()} {live_confidence:.1f}%",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, guess_color, 2)
        cv2.putText(frame, f"Round: {round_number}   Level: {level}",
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Player: {player_score}   AI: {ai_score}",
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Time left: {time_left:.1f}s   Win threshold: {win_threshold:.0f}%",
                    (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        if round_best_label == target_label and round_best_confidence >= win_threshold:
            cv2.putText(frame, "AI GUESSED IT", (frame.shape[1] // 2 - 150, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow(CHALLENGE_NAME, frame)

        if elapsed >= round_seconds:
            if round_best_label == target_label and round_best_confidence >= win_threshold:
                ai_score += 1
                result_text = f"AI +1 ({round_best_label} @ {round_best_confidence:.1f}%)"
            else:
                player_score += 1
                result_text = f"Player +1 (AI guessed {round_best_label or 'nothing'})"

            level = 1 + ((player_score + ai_score) // LEVEL_UP_EVERY)
            round_number += 1
            print(result_text)
            round_target = random.choice(round_emotions)
            round_start = time.time()
            round_best_label = None
            round_best_confidence = 0.0
            live_guess = "--"
            live_confidence = 0.0

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    print(f"\nFinal score - Player: {player_score}, AI: {ai_score}")


if __name__ == "__main__":
    main()
