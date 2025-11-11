"""
Utility functions for estimating planar homographies with DLT and RANSAC.

This module intentionally avoids calling OpenCV's findHomography so the core
algorithms stay transparent for learning and reporting purposes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class HomographyResult:
    """Container for RANSAC-estimated homography metadata."""

    H: np.ndarray
    inlier_mask: np.ndarray
    num_inliers: int
    num_matches: int
    reprojection_error: float

    @property
    def inlier_ratio(self) -> float:
        return float(self.num_inliers) / float(max(self.num_matches, 1))


def _prepare_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Points must be an array of shape (N, 2)")
    return pts


def normalize_points(points: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize 2D points using isotropic scaling for numerical stability.

    Parameters
    ----------
    points: Iterable of (x, y) coordinates.

    Returns
    -------
    normalized_points: np.ndarray of shape (N, 2)
        Points transformed by the normalization matrix.
    T: np.ndarray of shape (3, 3)
        Similarity transform matrix such that normalized_points = (T @ pts_homog)^T.
    """

    pts = _prepare_points(points)
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    if mean_dist <= 0:
        scale = 1.0
    else:
        scale = math.sqrt(2.0) / mean_dist

    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    normalized = (T @ pts_h.T).T
    return normalized[:, :2], T


def dlt_homography(src_pts: Sequence[Sequence[float]], dst_pts: Sequence[Sequence[float]]) -> np.ndarray:
    """Compute homography with the Direct Linear Transform (DLT) algorithm."""

    src = _prepare_points(src_pts)
    dst = _prepare_points(dst_pts)
    if src.shape[0] != dst.shape[0]:
        raise ValueError("Source and destination points must have the same length")
    if src.shape[0] < 4:
        raise ValueError("At least four point correspondences are required for DLT")

    src_norm, T_src = normalize_points(src)
    dst_norm, T_dst = normalize_points(dst)

    num_points = src.shape[0]
    A = np.zeros((num_points * 2, 9), dtype=float)
    for i in range(num_points):
        x, y = src_norm[i]
        u, v = dst_norm[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]

    _, _, vh = np.linalg.svd(A)
    H_norm = vh[-1].reshape(3, 3)

    # Denormalise: H = T_dst^{-1} * H_norm * T_src
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    H /= H[2, 2]
    return H


def project_points(H: np.ndarray, points: Sequence[Sequence[float]]) -> np.ndarray:
    """Project 2D points using a homography matrix.
    
    Points projected to infinity (w ≈ 0) are returned as NaN to avoid 
    division by zero errors.
    """

    pts = _prepare_points(points)
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    proj_h = (H @ pts_h.T).T
    
    # Avoid division by zero: use a small epsilon for near-zero denominators
    w = proj_h[:, 2:3]
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = proj_h[:, :2] / w
    
    # Mark points at infinity as NaN
    proj[np.abs(w.squeeze()) < 1e-10] = np.nan
    
    return proj


def reprojection_errors(H: np.ndarray, src_pts: Sequence[Sequence[float]], dst_pts: Sequence[Sequence[float]]) -> np.ndarray:
    """Compute Euclidean reprojection errors of src_pts mapped onto dst_pts.
    
    Points that project to infinity are assigned infinite error.
    """

    projected = project_points(H, src_pts)
    dst = _prepare_points(dst_pts)
    
    # Compute errors; NaN projections will result in NaN errors
    with np.errstate(invalid='ignore'):
        errors = np.linalg.norm(projected - dst, axis=1)
    
    # Replace NaN errors with infinity (these points are at infinity)
    errors = np.where(np.isnan(errors), np.inf, errors)
    
    return errors


def ransac_homography(
    src_pts: Sequence[Sequence[float]],
    dst_pts: Sequence[Sequence[float]],
    threshold: float = 3.0,
    max_iterations: int = 4000,
    confidence: float = 0.995,
    random_state: Optional[random.Random] = None,
) -> HomographyResult:
    """Robustly estimate a homography with RANSAC and DLT.

    Parameters
    ----------
    src_pts, dst_pts: collections of matched coordinates.
    threshold: maximum reprojection error (in pixels) for an inlier.
    max_iterations: hard limit on the number of RANSAC trials.
    confidence: desired probability that at least one random sample is free of outliers.
    random_state: optional random generator for reproducibility.
    """

    src = _prepare_points(src_pts)
    dst = _prepare_points(dst_pts)
    n_points = src.shape[0]
    if n_points != dst.shape[0]:
        raise ValueError("src_pts and dst_pts must contain the same number of points")
    if n_points < 4:
        raise ValueError("At least four correspondences are required for RANSAC")

    rng = random_state or random.Random()
    best_H: Optional[np.ndarray] = None
    best_inlier_mask = np.zeros(n_points, dtype=bool)
    best_error = math.inf

    sample_size = 4
    max_trials = max_iterations
    trial = 0

    while trial < max_trials:
        sample_indices = rng.sample(range(n_points), sample_size)
        try:
            H_candidate = dlt_homography(src[sample_indices], dst[sample_indices])
        except np.linalg.LinAlgError:
            trial += 1
            continue

        errors = reprojection_errors(H_candidate, src, dst)
        inliers = errors < threshold
        num_inliers = int(np.sum(inliers))

        if num_inliers >= sample_size:
            try:
                H_refined = dlt_homography(src[inliers], dst[inliers])
                # Check if matrix is invertible (determinant not too close to zero)
                det = np.linalg.det(H_refined)
                if abs(det) < 1e-10:
                    trial += 1
                    continue
                
                refined_errors = reprojection_errors(H_refined, src[inliers], dst[inliers])
                mean_error = float(np.mean(refined_errors)) if num_inliers > 0 else math.inf

                if num_inliers > best_inlier_mask.sum() or (
                    num_inliers == best_inlier_mask.sum() and mean_error < best_error
                ):
                    best_H = H_refined
                    best_inlier_mask = inliers
                    best_error = mean_error

                    inlier_ratio = num_inliers / n_points
                    if inlier_ratio > 0:
                        eps = max(1e-8, 1 - inlier_ratio ** sample_size)
                        dynamic_max = math.log(1 - confidence) / math.log(eps)
                        max_trials = min(max_iterations, int(math.ceil(dynamic_max)))
            except np.linalg.LinAlgError:
                trial += 1
                continue

        trial += 1

    if best_H is None:
        # Fallback: compute using all points despite failure to find consensus.
        try:
            best_H = dlt_homography(src, dst)
            # Verify the fallback homography is at least invertible
            det = np.linalg.det(best_H)
            if abs(det) < 1e-10:
                raise ValueError(
                    f"Failed to compute valid homography: determinant too small ({det:.2e}). "
                    f"Point correspondences may be degenerate or collinear."
                )
            best_inlier_mask = np.ones(n_points, dtype=bool)
            best_error = float(np.mean(reprojection_errors(best_H, src, dst)))
        except (np.linalg.LinAlgError, ValueError) as e:
            raise ValueError(
                f"RANSAC failed to find any valid homography after {max_iterations} iterations. "
                f"This typically indicates insufficient or poor quality matches."
            ) from e

    total_matches = n_points
    num_inliers = int(best_inlier_mask.sum())
    mean_error = best_error

    return HomographyResult(
        H=best_H,
        inlier_mask=best_inlier_mask,
        num_inliers=num_inliers,
        num_matches=total_matches,
        reprojection_error=mean_error,
    )
