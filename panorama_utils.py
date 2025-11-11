"""Panorama construction helpers built atop custom homography estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from feature_utils import detect_and_describe, match_descriptors, matches_to_points
from homography_utils import HomographyResult, ransac_homography


@dataclass
class PairwiseMatch:
    reference_image: np.ndarray
    target_image: np.ndarray
    keypoints_ref: List[cv2.KeyPoint]
    keypoints_target: List[cv2.KeyPoint]
    matches: List[cv2.DMatch]
    homography: HomographyResult
    reference_index: int = 0
    target_index: int = 0
    match_reference_mode: str = "reference"
    _target_to_reference_global: np.ndarray | None = field(default=None, repr=False)

    @property
    def src_points(self) -> np.ndarray:
        pts, _ = matches_to_points(self.keypoints_ref, self.keypoints_target, self.matches)
        return pts

    @property
    def dst_points(self) -> np.ndarray:
        _, pts = matches_to_points(self.keypoints_ref, self.keypoints_target, self.matches)
        return pts

    @property
    def target_to_reference(self) -> np.ndarray:
        """Invert the homography to map target → reference.
        
        Raises
        ------
        np.linalg.LinAlgError
            If the homography matrix is singular or near-singular.
        """
        try:
            H_inv = np.linalg.inv(self.homography.H)
            # Verify the inversion is reasonable
            if not np.isfinite(H_inv).all():
                raise np.linalg.LinAlgError("Inverted matrix contains non-finite values")
            return H_inv
        except np.linalg.LinAlgError as e:
            # Provide more context about the failure
            det = np.linalg.det(self.homography.H)
            raise np.linalg.LinAlgError(
                f"Cannot invert homography matrix (determinant={det:.2e}). "
                f"This likely indicates insufficient or poor quality matches. "
                f"Inliers: {self.homography.num_inliers}/{self.homography.num_matches}, "
                f"Ratio: {self.homography.inlier_ratio:.2%}"
            ) from e

    @property
    def target_to_reference_global(self) -> np.ndarray:
        """Return the homography mapping the target image directly to the global reference.

        Falls back to the local reference mapping when no global transform was supplied.
        """

        if self._target_to_reference_global is not None:
            return self._target_to_reference_global
        return self.target_to_reference

    def set_target_to_reference_global(self, H: np.ndarray) -> None:
        if H.ndim != 2 or H.shape != (3, 3):
            raise ValueError("Global homography must be a 3x3 matrix")
        if not np.isfinite(H).all():
            raise ValueError("Global homography contains non-finite values")
        if abs(H[2, 2]) < 1e-12:
            raise ValueError("Global homography has near-zero scale in homogeneous coordinate")
        H_norm = H / H[2, 2]
        self._target_to_reference_global = H_norm



def load_scene_images(scene_dir: Path) -> List[Path]:
    scene_dir = Path(scene_dir)
    image_paths = sorted(p for p in scene_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not image_paths:
        raise FileNotFoundError(f"No images found in {scene_dir}")
    return image_paths


def estimate_pairwise_homography(
    img_ref: np.ndarray,
    img_target: np.ndarray,
    method: str = "SIFT",
    ratio: float = 0.75,
    ransac_threshold: float = 3.0,
    max_ransac_iterations: int = 4000,
    confidence: float = 0.995,
    detector_kwargs: Dict | None = None,
) -> PairwiseMatch:
    detector_kwargs = detector_kwargs or {}
    kp_ref, desc_ref = detect_and_describe(img_ref, method=method, **detector_kwargs)
    kp_target, desc_target = detect_and_describe(img_target, method=method, **detector_kwargs)
    matches = match_descriptors(desc_ref, desc_target, method=method, ratio=ratio)
    if len(matches) < 4:
        raise RuntimeError(f"Insufficient matches ({len(matches)}) for homography estimation")

    src_pts, dst_pts = matches_to_points(kp_ref, kp_target, matches)
    homography = ransac_homography(
        src_pts,
        dst_pts,
        threshold=ransac_threshold,
        max_iterations=max_ransac_iterations,
        confidence=confidence,
    )
    return PairwiseMatch(
        reference_image=img_ref,
        target_image=img_target,
        keypoints_ref=kp_ref,
        keypoints_target=kp_target,
        matches=matches,
        homography=homography,
    )


def warp_image(
    image: np.ndarray,
    H: np.ndarray,
    output_shape: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    return cv2.warpPerspective(image, H, output_shape, flags=interpolation)


def compute_panorama_canvas(
    images: Sequence[np.ndarray],
    homographies_to_reference: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if len(images) != len(homographies_to_reference):
        raise ValueError("Number of images and homographies must match")

    corners_accum = []
    for img, H in zip(images, homographies_to_reference):
        h, w = img.shape[:2]
        corners = np.array(
            [[0, 0], [w, 0], [w, h], [0, h]],
            dtype=np.float32,
        )
        warped = cv2.perspectiveTransform(corners[None, :, :], H)[0]
        corners_accum.append(warped)

    all_corners = np.vstack(corners_accum)
    min_xy = np.floor(all_corners.min(axis=0)).astype(int)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int)

    width = int(max_xy[0] - min_xy[0])
    height = int(max_xy[1] - min_xy[1])

    translation = np.array(
        [[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return (width, height), translation


def linear_blend_panorama(
    images: Sequence[np.ndarray],
    homographies_to_reference: Sequence[np.ndarray],
    interpolation: int = cv2.INTER_LINEAR,
) -> Tuple[np.ndarray, np.ndarray]:
    canvas_size, translation = compute_panorama_canvas(images, homographies_to_reference)
    width, height = canvas_size

    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    weight_map = np.zeros((height, width, 1), dtype=np.float32)

    for img, H in zip(images, homographies_to_reference):
        H_shifted = translation @ H
        warped = cv2.warpPerspective(img, H_shifted, (width, height), flags=interpolation)
        mask = cv2.warpPerspective(
            np.ones(img.shape[:2], dtype=np.float32),
            H_shifted,
            (width, height),
        )
        mask = mask[..., None]
        accumulator += warped.astype(np.float32) * mask
        weight_map += mask

    weight_map[weight_map == 0] = 1.0
    panorama = accumulator / weight_map
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)
    return panorama, weight_map


def stitch_scene(
    image_paths: Sequence[Path | str],
    feature_method: str = "SIFT",
    ratio: float = 0.75,
    ransac_threshold: float = 3.0,
    blend_mode: str = "linear",
    detector_kwargs: Dict | None = None,
    match_strategy: str = "hybrid",
) -> Dict:
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in image_paths]
    if any(img is None for img in images):
        missing = [str(path) for img, path in zip(images, image_paths) if img is None]
        raise FileNotFoundError(f"Failed to read images: {missing}")

    reference = images[0]
    pairwise_results: List[PairwiseMatch] = []
    homographies = [np.eye(3)]

    detector_kwargs = detector_kwargs or {}
    strategy = match_strategy.lower()
    if strategy not in {"reference", "sequential", "hybrid"}:
        raise ValueError("match_strategy must be one of {'reference', 'sequential', 'hybrid'}")

    anchor_index = 0

    def _attempt_match(ref_idx: int, tgt_idx: int, mode_label: str) -> PairwiseMatch | None:
        ref_image = images[ref_idx]
        target_image = images[tgt_idx]
        try:
            result = estimate_pairwise_homography(
                ref_image,
                target_image,
                method=feature_method,
                ratio=ratio,
                ransac_threshold=ransac_threshold,
                detector_kwargs=detector_kwargs,
            )
        except (RuntimeError, np.linalg.LinAlgError, ValueError, cv2.error):
            return None

        result.reference_index = ref_idx
        result.target_index = tgt_idx
        result.match_reference_mode = mode_label

        try:
            local_H = result.target_to_reference
        except np.linalg.LinAlgError:
            return None

        global_H = homographies[ref_idx] @ local_H
        if not np.isfinite(global_H).all() or abs(global_H[2, 2]) < 1e-12:
            return None
        result.set_target_to_reference_global(global_H)
        return result

    for target_idx in range(1, len(images)):
        candidates: List[PairwiseMatch] = []

        if strategy in {"sequential", "hybrid"}:
            if anchor_index < target_idx:
                sequential_match = _attempt_match(anchor_index, target_idx, "sequential")
                if sequential_match is not None:
                    candidates.append(sequential_match)

        if strategy in {"reference", "hybrid"}:
            if 0 not in {c.reference_index for c in candidates}:
                reference_match = _attempt_match(0, target_idx, "reference")
                if reference_match is not None:
                    candidates.append(reference_match)

        if not candidates:
            img_name = Path(image_paths[target_idx]).name
            raise RuntimeError(
                f"Failed to compute homography for {img_name} (image {target_idx+1}/{len(images)})."
            )

        best_match = max(
            candidates,
            key=lambda match: (match.homography.num_inliers, -match.homography.reprojection_error),
        )

        homographies.append(best_match.target_to_reference_global.copy())
        pairwise_results.append(best_match)

        if strategy in {"sequential", "hybrid"}:
            anchor_index = target_idx

    if blend_mode.lower() == "linear":
        panorama, weight_map = linear_blend_panorama(images, homographies)
    else:
        raise ValueError(f"Unsupported blend mode: {blend_mode}")

    return {
        "panorama": panorama,
        "weight_map": weight_map,
        "pairwise_results": pairwise_results,
        "homographies": homographies,
        "images": images,
    }
