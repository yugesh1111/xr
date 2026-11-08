"""Video -> landmark array with MediaPipe Holistic (Tasks API).

Holistic finds the hands from the pose wrists instead of running a separate
palm detector on every frame, which is also what we want on the phone.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from . import layout

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task")
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "holistic_landmarker.task"


def ensure_model(path=DEFAULT_MODEL):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class LandmarkExtractor:
    def __init__(self, model_path=None):
        from mediapipe.tasks.python import BaseOptions, vision
        opts = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path or DEFAULT_MODEL))),
            running_mode=vision.RunningMode.VIDEO,
        )
        self._landmarker = vision.HolisticLandmarker.create_from_options(opts)
        self._last_ts = -1

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def frame(self, rgb, timestamp_ms):
        """One RGB frame -> (88, 3) array. x and z are rescaled by width/height
        so that all three axes share the same unit (fraction of image height)."""
        import mediapipe as mp
        h, w = rgb.shape[:2]
        ts = max(int(timestamp_ms), self._last_ts + 1)  # must be strictly increasing
        self._last_ts = ts
        res = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        out = np.full((layout.N_POINTS, 3), np.nan, dtype=np.float32)

        def put(block, lms, idx=None):
            if not lms:
                return
            pts = lms if idx is None else [lms[i] for i in idx]
            out[block] = [(p.x, p.y, p.z) for p in pts]

        put(layout.POSE, res.pose_landmarks)
        put(layout.LHAND, res.left_hand_landmarks)
        put(layout.RHAND, res.right_hand_landmarks)
        put(layout.FACE, res.face_landmarks, layout.FACE_IDX)
        out[:, [0, 2]] *= w / h
        return out


def read_video(path, frame_start=1, frame_end=-1, transform=None):
    """Yield (rgb_frame, timestamp_ms). frame_start/frame_end are 1-based and
    inclusive as in the WLASL metadata; frame_end=-1 means until the end.
    `transform(rgb) -> rgb` is applied to every frame (used for robustness tests)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            i += 1
            if i < frame_start:
                continue
            if frame_end > 0 and i > frame_end:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            yield (transform(rgb) if transform else rgb), 1000.0 * i / fps
    finally:
        cap.release()


def extract_video(path, frame_start=1, frame_end=-1, model_path=None, transform=None):
    """Return (T, 88, 3) float32 array for a clip.

    A fresh landmarker per clip: in VIDEO mode MediaPipe tracks across frames,
    and we don't want tracking state leaking from one clip into the next."""
    with LandmarkExtractor(model_path) as ex:
        frames = [ex.frame(rgb, ts) for rgb, ts in read_video(path, frame_start, frame_end, transform)]
    if not frames:
        return np.zeros((0, layout.N_POINTS, 3), dtype=np.float32)
    return np.stack(frames)
