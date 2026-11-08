"""Landmark layout shared by every data source (our extractor, Kaggle ASL Signs).

One frame = 88 points: 33 pose + 21 left hand + 21 right hand + 13 face points.
Coordinates are (x, y, z); a missing point is NaN.
"""
import numpy as np

N_POSE, N_HAND = 33, 21

# Small face subset: enough for "where is the hand relative to the face"
# and mouth shape, without carrying 468 points around.
FACE_IDX = [1, 10, 152, 33, 263, 61, 291, 0, 17, 13, 14, 78, 308]

POSE = slice(0, 33)
LHAND = slice(33, 54)
RHAND = slice(54, 75)
FACE = slice(75, 75 + len(FACE_IDX))
N_POINTS = 75 + len(FACE_IDX)

# pose indices (MediaPipe numbering)
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16

# face points, as offsets into the FACE block
F_NOSE, F_FOREHEAD, F_CHIN = 0, 1, 2
F_UPPER_LIP, F_LOWER_LIP = 9, 10
F_MOUTH_L, F_MOUTH_R = 5, 6

# hand indices
WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17
TIPS = [4, 8, 12, 16, 20]
# (a, b, c) triples: flex angle at b
FINGER_JOINTS = [
    (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
    (0, 1, 2),
]

POSE_LR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
                 (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
FACE_LR_PAIRS = [(3, 4), (5, 6), (11, 12)]


def mirror_permutation():
    """Index permutation that swaps left/right body parts (used with x -> -x)."""
    perm = np.arange(N_POINTS)
    for a, b in POSE_LR_PAIRS:
        perm[a], perm[b] = b, a
    perm[LHAND], perm[RHAND] = np.arange(54, 75), np.arange(33, 54)
    for a, b in FACE_LR_PAIRS:
        perm[75 + a], perm[75 + b] = 75 + b, 75 + a
    return perm
