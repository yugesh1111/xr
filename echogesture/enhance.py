"""Adaptive sharpening before MediaPipe (part of the deployed pipeline).

Blur is the corruption that hurts most: MediaPipe still finds the body, but
the finger landmarks come out wrong, so handshape - the most informative
feature group - breaks. Sharpening blurry frames recovers much of it.

Sharpening every frame costs accuracy on good video, and sharpening a dark
frame amplifies sensor noise (dark accuracy fell 4.7 points when we tried).
So a frame is sharpened only when it is blurry AND not dark:

    sharpness = variance of the Laplacian of the grey image, resized to 256 px high
    sharpen if sharpness < 35 and mean grey level >= 50

On a random sample of WLASL frames, no clean frame scored below 30, while
the median blurred frame scored 20 - so clean video is almost never touched.
The Android app (android/.../Enhance.kt) implements the same rule.
"""
import cv2
import numpy as np

SHARPNESS_MAX = 35.0   # below this a frame counts as blurry
MIN_LIGHT = 50.0       # below this mean grey level a frame counts as dark: leave it alone
AMOUNT, SIGMA = 2.0, 2.0


def sharpness(grey):
    h, w = grey.shape
    small = cv2.resize(grey, (max(int(w * 256 / h), 1), 256), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def unsharp(rgb, amount=AMOUNT, sigma=SIGMA):
    soft = cv2.GaussianBlur(rgb, (0, 0), sigma)
    return cv2.addWeighted(rgb, 1 + amount, soft, -amount, 0)


def needs_sharpening(rgb):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return grey.mean() >= MIN_LIGHT and sharpness(grey) < SHARPNESS_MAX


def enhance(rgb):
    """RGB uint8 frame -> frame to give MediaPipe."""
    return unsharp(rgb) if needs_sharpening(rgb) else rgb
