"""Feature detection and matching utilities for homography estimation."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


def create_detector(method: str, **kwargs) -> cv2.Feature2D:
    """Factory for local feature detectors/descriptors."""

    method_upper = method.upper()
    if method_upper == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("SIFT is unavailable. Ensure opencv-contrib-python is installed.")
        return cv2.SIFT_create(**kwargs)
    if method_upper == "SURF":
        if not hasattr(cv2, "xfeatures2d") or not hasattr(cv2.xfeatures2d, "SURF_create"):
            raise RuntimeError(
                "SURF is unavailable in this OpenCV build. "
                "Install opencv-contrib-python and enable nonfree features."
            )
        try:
            return cv2.xfeatures2d.SURF_create(**kwargs)
        except cv2.error as err:
            raise RuntimeError(
                "SURF is unavailable in this OpenCV configuration. "
                "Rebuild OpenCV with OPENCV_ENABLE_NONFREE to use SURF."
            ) from err
    if method_upper == "ORB":
        return cv2.ORB_create(**kwargs)

    raise ValueError(f"Unsupported feature method: {method}")


def detect_and_describe(
    image: np.ndarray,
    method: str = "SIFT",
    mask: np.ndarray | None = None,
    **kwargs,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Detect keypoints and compute descriptors for an image."""

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    detector = create_detector(method, **kwargs)
    keypoints, descriptors = detector.detectAndCompute(gray, mask)
    if descriptors is None or len(keypoints) == 0:
        raise RuntimeError(f"No descriptors found using method {method}")
    return keypoints, descriptors


def _matcher_norm(method: str) -> int:
    if method.upper() in {"SIFT", "SURF"}:
        return cv2.NORM_L2
    return cv2.NORM_HAMMING


def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    method: str,
    ratio: float = 0.75,
    cross_check: bool = False,
) -> List[cv2.DMatch]:
    """Match descriptors with Lowe's ratio test."""

    norm_type = _matcher_norm(method)
    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    knn = matcher.knnMatch(desc1, desc2, k=2)
    good_matches: List[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    if cross_check:
        matcher_inv = cv2.BFMatcher(norm_type, crossCheck=False)
        knn_inv = matcher_inv.knnMatch(desc2, desc1, k=2)
        reciprocal = {m.queryIdx: m.trainIdx for m in good_matches}
        keep_indices = set()
        for pair in knn_inv:
            if len(pair) < 2:
                continue
            m_inv, n_inv = pair
            if m_inv.distance < ratio * n_inv.distance:
                forward_match = reciprocal.get(m_inv.trainIdx)
                if forward_match is not None and forward_match == m_inv.queryIdx:
                    keep_indices.add(m_inv.trainIdx)
        good_matches = [m for m in good_matches if m.queryIdx in keep_indices]

    good_matches.sort(key=lambda m: m.distance)
    return good_matches


def matches_to_points(
    kp1: Sequence[cv2.KeyPoint],
    kp2: Sequence[cv2.KeyPoint],
    matches: Iterable[cv2.DMatch],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert CV matches to coordinate arrays."""

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    return src_pts, dst_pts


def draw_keypoints(image: np.ndarray, keypoints: Sequence[cv2.KeyPoint], color=(0, 255, 0)) -> np.ndarray:
    return cv2.drawKeypoints(image, list(keypoints), None, color=color, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def draw_matches(
    img1: np.ndarray,
    kp1: Sequence[cv2.KeyPoint],
    img2: np.ndarray,
    kp2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    inliers_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Visualize matches with optional inlier mask."""

    if inliers_mask is None:
        matches_to_draw = matches
    else:
        matches_to_draw = [m for m, keep in zip(matches, inliers_mask) if keep]
    return cv2.drawMatches(
        img1,
        list(kp1),
        img2,
        list(kp2),
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
