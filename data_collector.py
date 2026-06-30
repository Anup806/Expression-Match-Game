"""
data_collector.py

STEP 1 of the pipeline. Run this first.

Shows your webcam feed with a target emotion prompt. When you act out the
emotion, press SPACE to save that frame's geometric features as a labeled
training sample. Samples are appended to data/dataset.csv.

Controls:
    [SPACE] -> capture current frame as a sample for the active emotion
    [N]     -> move to next emotion
    [Q]     -> quit and save

Aim for at least 40-60 samples per emotion. Vary your head angle, distance
from camera, and intensity slightly between captures -- a model trained on
identical repeated poses overfits and breaks the moment you move.
"""

import os
import cv2
import pandas as pd

from challenge_config import CHALLENGE_EMOTIONS, DATA_DIR, CSV_PATH
from feature_extraction import extract_features, FEATURE_NAMES
from mediapipe_compat import create_face_landmarker, detect_face_landmarks, draw_face_mesh
from mediapipe.tasks.python import vision as mp_vision


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    face_landmarker = create_face_landmarker()
    drawing_utils = mp_vision.drawing_utils
    drawing_styles = mp_vision.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check it isn't in use by another app.")

    rows = []
    expr_idx = 0
    sample_count = 0

    print("Data collection started. SPACE=capture  N=next expression  Q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        results = detect_face_landmarks(face_landmarker, frame)

        label, display_name, emoji = CHALLENGE_EMOTIONS[expr_idx]

        cv2.putText(frame, f"Act this emotion: {display_name.upper()} {emoji}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples collected for '{label}': {sample_count}",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "SPACE=capture  N=next emotion  Q=quit",
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        face_detected = bool(results.multi_face_landmarks)
        if face_detected:
            landmarks = results.multi_face_landmarks[0].landmark
            draw_face_mesh(frame, results.multi_face_landmarks[0], drawing_utils, drawing_styles)
        else:
            cv2.putText(frame, "No face detected", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Expression Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and face_detected:
            features = extract_features(landmarks)
            rows.append(list(features) + [label])
            sample_count += 1
            print(f"Captured sample #{sample_count} for '{label}'")

        elif key == ord('n'):
            expr_idx = (expr_idx + 1) % len(CHALLENGE_EMOTIONS)
            sample_count = 0

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()

    if rows:
        df_new = pd.DataFrame(rows, columns=FEATURE_NAMES + ["label"])
        if os.path.exists(CSV_PATH):
            df_old = pd.read_csv(CSV_PATH)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_final = df_new
        df_final.to_csv(CSV_PATH, index=False)
        print(f"\nSaved {len(rows)} new samples. Total dataset size: {len(df_final)} -> {CSV_PATH}")
        print(df_final["label"].value_counts())
    else:
        print("No samples captured.")


if __name__ == "__main__":
    main()
