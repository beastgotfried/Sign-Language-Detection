"""Shared feature engineering for SignBridge.

This module converts MediaPipe hand landmarks into a fixed 48-dimensional
feature vector. The layout is:
- 22 features for hand 1 (8 distances + 6 angles + 5 curl + 3 position)
- 22 features for hand 2 (same as hand 1, or zeros if only one hand)
- 4 cross-hand features (or zeros if only one hand)

The implementation accepts a single hand or two hands and always returns the
same vector length so training and inference stay aligned.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from utils.math_utils import angle, distance

FEATURE_DIMENSIONS = 48
LANDMARKS_PER_HAND = 21
PER_HAND_FEATURES = 22
CROSS_HAND_FEATURES = 4


def _points_from_flat(flat_landmarks: Sequence[float]) -> np.ndarray:
    """Convert flat 63-value list to (21, 3) point array."""
    values = np.asarray(flat_landmarks, dtype=float)
    if values.size != LANDMARKS_PER_HAND * 3:
        return np.empty((0, 3), dtype=float)
    return values.reshape(LANDMARKS_PER_HAND, 3)


def _is_landmark_object(value: object) -> bool:
    """Check if value has MediaPipe landmark attributes."""
    return hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z")


def _points_from_landmark_objects(landmarks: Sequence[object]) -> np.ndarray:
    """Convert MediaPipe landmark objects to (21, 3) point array."""
    if len(landmarks) != LANDMARKS_PER_HAND:
        return np.empty((0, 3), dtype=float)
    points = []
    for landmark in landmarks:
        points.append([float(landmark.x), float(landmark.y), float(landmark.z)])
    return np.asarray(points, dtype=float)


def _coerce_hand_points(landmarks: object) -> list[np.ndarray]:
    """Coerce landmarks into a list of (21, 3) hand arrays."""
    if landmarks is None:
        return []

    # Handle flat list (63 values)
    if isinstance(landmarks, Sequence) and not isinstance(landmarks, (str, bytes)):
        if len(landmarks) == LANDMARKS_PER_HAND * 3 and all(
            isinstance(v, (int, float, np.floating, np.integer)) for v in landmarks
        ):
            points = _points_from_flat(landmarks)
            return [points] if points.size else []

    # Handle numpy array
    if isinstance(landmarks, np.ndarray):
        if landmarks.ndim == 2 and landmarks.shape == (LANDMARKS_PER_HAND, 3):
            return [landmarks.astype(float, copy=False)]
        if landmarks.ndim == 1 and landmarks.size == LANDMARKS_PER_HAND * 3:
            points = _points_from_flat(landmarks.tolist())
            return [points] if points.size else []

    # Handle MediaPipe landmark objects
    if isinstance(landmarks, Sequence) and not isinstance(landmarks, (str, bytes)):
        if not landmarks:
            return []
        first_item = landmarks[0]
        if _is_landmark_object(first_item):
            points = _points_from_landmark_objects(landmarks)
            return [points] if points.size else []

    return []


def _hand_span(points: np.ndarray) -> float:
    """Compute bounding box span for normalization."""
    if points.size == 0:
        return 1.0
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = float(np.max(maxs - mins))
    return span if span > 0 else 1.0


def normalise_landmarks(landmarks: object) -> np.ndarray:
    """Translate hand landmarks so wrist is at origin and scale by hand size."""
    hand_points_list = _coerce_hand_points(landmarks)
    if not hand_points_list:
        return np.zeros((LANDMARKS_PER_HAND, 3), dtype=float)

    points = hand_points_list[0].astype(float, copy=False)
    wrist = points[0]
    translated = points - wrist
    scale = _hand_span(points)
    return translated / scale


def compute_distances(landmarks: object) -> np.ndarray:
    """Compute 8 distance features: 5 fingertip-to-wrist + 3 fingertip-to-fingertip."""
    points = normalise_landmarks(landmarks)
    if points.shape != (LANDMARKS_PER_HAND, 3):
        return np.zeros(8, dtype=float)

    wrist = points[0]
    fingertips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

    # 5 fingertip-to-wrist distances
    features = [distance(points[tip], wrist) for tip in fingertips]

    # 3 inter-fingertip distances
    features.extend([
        distance(points[4], points[8]),    # thumb to index
        distance(points[8], points[12]),   # index to middle
        distance(points[12], points[16]),  # middle to ring
    ])

    return np.asarray(features, dtype=float)


def compute_angles(landmarks: object) -> np.ndarray:
    """Compute 6 angle features: finger angles and inter-finger angles."""
    points = normalise_landmarks(landmarks)
    if points.shape != (LANDMARKS_PER_HAND, 3):
        return np.zeros(6, dtype=float)

    wrist = points[0]

    angles_list = [
        # 5 intra-finger angles (each finger's 3-point angle)
        angle(points[1], points[2], points[3]),    # thumb
        angle(points[5], points[6], points[7]),    # index
        angle(points[9], points[10], points[11]),  # middle
        angle(points[13], points[14], points[15]), # ring
        angle(points[17], points[18], points[19]), # pinky
        # 1 inter-finger angle (between index and middle at wrist)
        angle(points[8], wrist, points[12]),
    ]

    return np.asarray(angles_list, dtype=float)


def compute_curl(landmarks: object) -> np.ndarray:
    """Compute 5 curl/flexion features: one per finger."""
    points = normalise_landmarks(landmarks)
    if points.shape != (LANDMARKS_PER_HAND, 3):
        return np.zeros(5, dtype=float)

    # Finger structure: (base, mid1, mid2, tip)
    finger_groups = [
        (1, 2, 3, 4),      # thumb
        (5, 6, 7, 8),      # index
        (9, 10, 11, 12),   # middle
        (13, 14, 15, 16),  # ring
        (17, 18, 19, 20),  # pinky
    ]

    curl_scores = []
    for base, mid1, mid2, tip in finger_groups:
        # Curl = ratio of base-to-tip distance vs sum of joint lengths
        straight_distance = distance(points[base], points[tip])
        bent_distance = (
            distance(points[base], points[mid1]) +
            distance(points[mid1], points[mid2]) +
            distance(points[mid2], points[tip])
        )
        if bent_distance == 0:
            curl_scores.append(0.0)
        else:
            curl_scores.append(straight_distance / bent_distance)

    return np.asarray(curl_scores, dtype=float)


def compute_position(landmarks: object) -> np.ndarray:
    """Compute 3 position features: normalized wrist coordinates."""
    hand_points_list = _coerce_hand_points(landmarks)
    if not hand_points_list:
        return np.zeros(3, dtype=float)

    points = hand_points_list[0]
    wrist = points[0]
    mins = points.min(axis=0)
    span = points.max(axis=0) - mins
    span[span == 0] = 1.0
    position = (wrist - mins) / span

    return np.asarray(position, dtype=float)


def _hand_block_features(hand_landmarks: object) -> np.ndarray:
    """Compute 22 features for a single hand."""
    distances = compute_distances(hand_landmarks)
    angles = compute_angles(hand_landmarks)
    curls = compute_curl(hand_landmarks)
    position = compute_position(hand_landmarks)
    return np.concatenate([distances, angles, curls, position]).astype(float)


def _cross_hand_features(hand_a: np.ndarray, hand_b: np.ndarray) -> np.ndarray:
    """Compute 4 cross-hand features from two hand point arrays."""
    # Compute on normalized hands
    norm_a = normalise_landmarks(hand_a)
    norm_b = normalise_landmarks(hand_b)

    if norm_a.shape != (LANDMARKS_PER_HAND, 3) or norm_b.shape != (LANDMARKS_PER_HAND, 3):
        return np.zeros(CROSS_HAND_FEATURES, dtype=float)

    wrist_dist = distance(norm_a[0], norm_b[0])
    thumb_dist = distance(norm_a[4], norm_b[4])
    index_dist = distance(norm_a[8], norm_b[8])
    fingertip_avg = float(np.mean([
        distance(norm_a[4], norm_b[4]),
        distance(norm_a[8], norm_b[8]),
        distance(norm_a[12], norm_b[12]),
        distance(norm_a[16], norm_b[16]),
        distance(norm_a[20], norm_b[20]),
    ]))

    return np.asarray([wrist_dist, thumb_dist, index_dist, fingertip_avg], dtype=float)


def extract_features(landmarks: object) -> np.ndarray:
    """Return a fixed 48-dimensional feature vector for one or two hands."""
    hand_points_list = _coerce_hand_points(landmarks)

    if not hand_points_list:
        return np.zeros(FEATURE_DIMENSIONS, dtype=float)

    # First hand features
    first_hand = _hand_block_features(hand_points_list[0])

    # Second hand features (or zeros)
    second_hand = (
        _hand_block_features(hand_points_list[1])
        if len(hand_points_list) > 1
        else np.zeros(PER_HAND_FEATURES, dtype=float)
    )

    # Cross-hand features (or zeros if only one hand)
    cross_hand = (
        _cross_hand_features(hand_points_list[0], hand_points_list[1])
        if len(hand_points_list) > 1
        else np.zeros(CROSS_HAND_FEATURES, dtype=float)
    )

    # Concatenate all features
    features = np.concatenate([first_hand, second_hand, cross_hand]).astype(float)

    # Ensure we return exactly FEATURE_DIMENSIONS
    if features.size != FEATURE_DIMENSIONS:
        padded = np.zeros(FEATURE_DIMENSIONS, dtype=float)
        padded[: min(features.size, FEATURE_DIMENSIONS)] = features[:FEATURE_DIMENSIONS]
        return padded

    return features


def landmarks_to_vector(hand_landmarks: object) -> list[float]:
    """Flatten MediaPipe landmarks into a legacy 63-value list."""
    hand_points = _coerce_hand_points(hand_landmarks)
    if not hand_points:
        return []
    return hand_points[0].reshape(-1).astype(float).tolist()