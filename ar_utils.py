"""Augmented Reality compositing utilities using custom homography estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from feature_utils import detect_and_describe, match_descriptors, matches_to_points
from homography_utils import HomographyResult, ransac_homography


@dataclass
class FrameResult:
    index: int
    homography: Optional[HomographyResult]
    matches_kept: int
    matches_total: int
    inlier_ratio: float
    frame: np.ndarray
    augmented_frame: np.ndarray
    used_previous: bool = False


def _central_crop_to_aspect(image: np.ndarray, target_aspect: float) -> np.ndarray:
    h, w = image.shape[:2]
    current_aspect = w / h
    if abs(current_aspect - target_aspect) < 1e-3:
        return image
    if current_aspect > target_aspect:
        # Too wide, crop horizontally
        new_w = int(target_aspect * h)
        start = (w - new_w) // 2
        return image[:, start : start + new_w]
    else:
        # Too tall, crop vertically
        new_h = int(w / target_aspect)
        start = (h - new_h) // 2
        return image[start : start + new_h, :]


def _resize_to(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def estimate_homography_for_frame(
    cover_keypoints: List[cv2.KeyPoint],
    cover_descriptors: np.ndarray,
    frame: np.ndarray,
    method: str = "SIFT",
    ratio: float = 0.75,
    ransac_threshold: float = 4.0,
    max_ransac_iterations: int = 4000,
    detector_kwargs: Dict | None = None,
) -> Tuple[Optional[HomographyResult], int, int]:
    detector_kwargs = detector_kwargs or {}
    try:
        kp_frame, desc_frame = detect_and_describe(frame, method=method, **detector_kwargs)
    except RuntimeError:
        return None, 0, 0

    matches = match_descriptors(cover_descriptors, desc_frame, method=method, ratio=ratio)
    total_matches = len(matches)
    if total_matches < 4:
        return None, 0, total_matches

    src_pts, dst_pts = matches_to_points(cover_keypoints, kp_frame, matches)
    try:
        homography = ransac_homography(
            src_pts,
            dst_pts,
            threshold=ransac_threshold,
            max_iterations=max_ransac_iterations,
        )
    except Exception:
        return None, 0, total_matches
    inliers = int(homography.num_inliers)
    return homography, inliers, total_matches


def composite_ar_video(
    cover_image_path: Path | str,
    scene_video_path: Path | str,
    source_video_path: Path | str,
    output_path: Path | str,
    method: str = "SIFT",
    ratio: float = 0.75,
    ransac_threshold: float = 4.0,
    detector_kwargs: Dict | None = None,
    frame_step: int = 1,
    max_frames: Optional[int] = None,
    fallback_to_previous: bool = True,
) -> List[FrameResult]:
    cover = cv2.imread(str(cover_image_path), cv2.IMREAD_COLOR)
    if cover is None:
        raise FileNotFoundError(f"Failed to load reference image: {cover_image_path}")

    scene_cap = cv2.VideoCapture(str(scene_video_path))
    source_cap = cv2.VideoCapture(str(source_video_path))
    if not scene_cap.isOpened():
        raise FileNotFoundError(f"Cannot open target video: {scene_video_path}")
    if not source_cap.isOpened():
        raise FileNotFoundError(f"Cannot open source video: {source_video_path}")

    cover_h, cover_w = cover.shape[:2]
    cover_aspect = cover_w / cover_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = scene_cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(scene_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(scene_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), fourcc, fps / frame_step, (width, height))

    detector_kwargs = detector_kwargs or {}
    kp_cover, desc_cover = detect_and_describe(cover, method=method, **detector_kwargs)
    frame_results: List[FrameResult] = []
    previous_homography: Optional[HomographyResult] = None
    frame_index = 0
    source_index = 0

    while True:
        ret, frame = scene_cap.read()
        if not ret:
            break
        if frame_step > 1 and frame_index % frame_step != 0:
            frame_index += 1
            continue

        if max_frames is not None and len(frame_results) >= max_frames:
            break

        source_ret, source_frame = source_cap.read()
        if not source_ret:
            source_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            source_index = 0
            source_ret, source_frame = source_cap.read()
            if not source_ret:
                raise RuntimeError("Source video has no frames to use for AR overlay")
        source_index += 1

        cropped_source = _central_crop_to_aspect(source_frame, cover_aspect)
        resized_source = _resize_to(cropped_source, (cover_w, cover_h))

        homography, inliers, total_matches = estimate_homography_for_frame(
            kp_cover,
            desc_cover,
            frame,
            method=method,
            ratio=ratio,
            ransac_threshold=ransac_threshold,
            detector_kwargs=detector_kwargs,
        )

        used_previous = False
        if homography is None and fallback_to_previous and previous_homography is not None:
            homography = previous_homography
            inliers = previous_homography.num_inliers
            total_matches = previous_homography.num_matches
            used_previous = True
        if homography is None:
            augmented = frame.copy()
            frame_results.append(
                FrameResult(
                    index=frame_index,
                    homography=None,
                    matches_kept=0,
                    matches_total=total_matches,
                    inlier_ratio=0.0,
                    frame=frame,
                    augmented_frame=augmented,
                    used_previous=used_previous,
                )
            )
            writer.write(augmented)
            frame_index += 1
            continue

        h_matrix = homography.H
        warp = cv2.warpPerspective(resized_source, h_matrix, (width, height))
        mask = cv2.warpPerspective(
            np.ones((cover_h, cover_w), dtype=np.float32),
            h_matrix,
            (width, height),
        )
        mask = np.clip(mask, 0, 1)
        blurred_mask = cv2.GaussianBlur(mask, (15, 15), sigmaX=5)
        blurred_mask = np.clip(blurred_mask, 0, 1)
        # Ensure mask has 3 channels to broadcast with RGB images
        blurred_mask = blurred_mask[..., None]

        warped_rgb = warp.astype(np.float32)
        frame_float = frame.astype(np.float32)
        augmented = (blurred_mask * warped_rgb + (1 - blurred_mask) * frame_float).astype(np.uint8)

        writer.write(augmented)
        frame_results.append(
            FrameResult(
                index=frame_index,
                homography=homography,
                matches_kept=inliers,
                matches_total=total_matches,
                inlier_ratio=homography.inlier_ratio if homography else 0.0,
                frame=frame,
                augmented_frame=augmented,
                used_previous=used_previous,
            )
        )
        previous_homography = homography
        frame_index += 1

    scene_cap.release()
    source_cap.release()
    writer.release()

    return frame_results
