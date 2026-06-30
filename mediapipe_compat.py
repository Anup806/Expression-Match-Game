"""
mediapipe_compat.py

Compatibility helpers for newer MediaPipe releases where `mp.solutions`
is no longer exposed at the top level.
"""

import os
import types
import urllib.request

import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core import image as image_lib


FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = os.path.join("data", "face_landmarker.task")


def ensure_face_landmarker_model(model_path: str = DEFAULT_MODEL_PATH) -> str:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, model_path)
    return model_path


def create_face_landmarker(model_path: str = DEFAULT_MODEL_PATH):
    model_path = ensure_face_landmarker_model(model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.FaceLandmarker.create_from_options(options)


def detect_face_landmarks(face_landmarker, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = image_lib.Image(image_lib.ImageFormat.SRGB, rgb)
    result = face_landmarker.detect(mp_image)
    landmarks = [types.SimpleNamespace(landmark=face) for face in result.face_landmarks]
    return types.SimpleNamespace(multi_face_landmarks=landmarks)


def draw_face_mesh(frame_bgr, face_landmarks, drawing_utils, drawing_styles):
    landmark_list = getattr(face_landmarks, "landmark", face_landmarks)
    drawing_utils.draw_landmarks(
        frame_bgr,
        landmark_list,
        vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
    )