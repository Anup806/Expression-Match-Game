"""
feature_extraction.py

Converts MediaPipe Face Mesh landmarks into a fixed-length numeric feature
vector describing facial expression geometry (mouth shape, eye openness,
eyebrow position, jaw drop).

These features are hand-engineered, not deep-learned. That is intentional:
it keeps the project lightweight, runs in real time on CPU (no GPU needed
at all for this project), and is fully explainable -- you can point at any
number in the vector and say exactly what it measures and why. That is a
much stronger thing to say in an interview than "I called DeepFace.predict()".
"""

import numpy as np

# MediaPipe Face Mesh landmark indices used for feature extraction.
# Full 468-point map: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
LM = {
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "left_eye_left": 33,
    "left_eye_right": 133,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "right_eye_left": 362,
    "right_eye_right": 263,
    "left_eyebrow_inner": 105,
    "right_eyebrow_inner": 334,
    "nose_tip": 1,
    "chin": 152,
    "left_cheek": 234,
    "right_cheek": 454,
}

FEATURE_NAMES = [
    "mouth_width",
    "mouth_open",
    "mouth_corner_lift",
    "left_eye_open",
    "right_eye_open",
    "left_eyebrow_raise",
    "right_eyebrow_raise",
    "eyebrow_furrow",
    "jaw_drop",
]


def _xy(landmarks, idx):
    p = landmarks[idx]
    return np.array([p.x, p.y])


def _dist(landmarks, i, j):
    return float(np.linalg.norm(_xy(landmarks, i) - _xy(landmarks, j)))


def extract_features(landmarks):
    """
    landmarks: the .landmark list from a MediaPipe FaceMesh result
               i.e. results.multi_face_landmarks[0].landmark

    Returns: np.float32 array of shape (len(FEATURE_NAMES),)
    """
    # Normalize every distance by face width so features stay roughly
    # constant regardless of how close the user sits to the camera.
    face_width = _dist(landmarks, LM["left_cheek"], LM["right_cheek"])
    if face_width < 1e-6:
        face_width = 1e-6

    mouth_width = _dist(landmarks, LM["mouth_left"], LM["mouth_right"]) / face_width
    mouth_open = _dist(landmarks, LM["mouth_top"], LM["mouth_bottom"]) / face_width

    # Mouth corner lift: positive when corners sit higher (smaller y) than
    # the mouth center -> smile/laugh. Negative -> frown/cry.
    mouth_center_y = (landmarks[LM["mouth_top"]].y + landmarks[LM["mouth_bottom"]].y) / 2
    corner_avg_y = (landmarks[LM["mouth_left"]].y + landmarks[LM["mouth_right"]].y) / 2
    mouth_corner_lift = (mouth_center_y - corner_avg_y) / face_width

    left_eye_open = _dist(landmarks, LM["left_eye_top"], LM["left_eye_bottom"]) / face_width
    right_eye_open = _dist(landmarks, LM["right_eye_top"], LM["right_eye_bottom"]) / face_width

    left_eyebrow_raise = _dist(landmarks, LM["left_eyebrow_inner"], LM["left_eye_top"]) / face_width
    right_eyebrow_raise = _dist(landmarks, LM["right_eyebrow_inner"], LM["right_eye_top"]) / face_width

    # Eyebrows pulled together -> tension / concentration signal
    eyebrow_furrow = _dist(landmarks, LM["left_eyebrow_inner"], LM["right_eyebrow_inner"]) / face_width

    jaw_drop = _dist(landmarks, LM["chin"], LM["nose_tip"]) / face_width

    return np.array([
        mouth_width,
        mouth_open,
        mouth_corner_lift,
        left_eye_open,
        right_eye_open,
        left_eyebrow_raise,
        right_eyebrow_raise,
        eyebrow_furrow,
        jaw_drop,
    ], dtype=np.float32)
