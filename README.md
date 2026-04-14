# Homography Estimation

A complete computer vision pipeline for planar homography estimation, panorama stitching, and augmented reality compositing — implemented from scratch in Python without relying on high-level wrappers.

## Overview

This project covers three interconnected tasks:

| Task | Description |
|------|-------------|
| **Homography Estimation** | Compute the 3×3 homography matrix from point correspondences using DLT + RANSAC |
| **Panorama Stitching** | Warp and blend multi-image sequences into seamless panoramic mosaics |
| **Augmented Reality** | Overlay a source video onto a planar surface tracked frame-by-frame |

---

## Demo

| Panorama (graffiti scene) | AR compositing |
|---|---|
| ![graffiti panorama](panorama_results/v_graffiti_panorama.png) | See `ar_dynamic_result.mp4` |

---

## Features

- **DLT (Direct Linear Transform)** — manual implementation with point normalization for numerical stability
- **RANSAC** — robust homography estimation with adaptive iteration count and configurable inlier threshold
- **Feature detection** — SIFT and ORB support via OpenCV, with Lowe's ratio test + cross-check filtering
- **Panorama construction** — sequential / reference / hybrid matching strategies with linear alpha blending
- **AR compositing** — per-frame homography estimation with temporal fallback and Gaussian-feathered blending
- **Evaluation** — reprojection error statistics and inlier-ratio metrics across 6 scenes

---

## Project Structure

```
.
├── homography_estimation.ipynb   # Main notebook (pipeline + results)
├── panorama_dataset/             # Input image sequences (6 scenes × 6 images)
│   ├── v_bird/
│   ├── v_boat/
│   ├── v_circus/
│   ├── v_graffiti/
│   ├── v_soldiers/
│   └── v_weapons/
├── panorama_results/             # Output panoramic images
├── ar_dataset/
│   ├── book.mov                  # Scene video
│   ├── ar_source.mov             # Content to overlay
│   └── cv_cover.jpg             # Reference image for tracking
├── ar_dynamic_result.mp4         # AR output video
└── demo.mp4                      # Full demo video
```

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the notebook

```bash
jupyter notebook homography_estimation.ipynb
```

Run cells top-to-bottom. The notebook is self-contained — it loads datasets, runs the pipeline, and displays results inline.

---

## Implementation Details

### Homography Estimation (DLT + RANSAC)

The Direct Linear Transform builds a system of linear equations from point correspondences and solves for the homography via SVD. Points are normalized before the solve to improve numerical conditioning.

RANSAC wraps the DLT to robustly handle outliers:
1. Randomly sample 4 point pairs
2. Estimate H via DLT
3. Count inliers (reprojection error < threshold)
4. Repeat and keep the best H

### Panorama Stitching

1. Detect and match features between image pairs
2. Estimate pairwise homographies via RANSAC
3. Compose global homographies relative to a reference image
4. Warp all images onto a shared canvas
5. Blend overlapping regions with linear alpha blending

### Augmented Reality

1. Pre-compute descriptors on the reference cover image
2. For each frame: detect features, match against reference, estimate H via RANSAC
3. Warp the source video frame using the estimated H
4. Composite onto the scene frame with feathered blending
5. Fall back to the previous frame's H when matching fails

---

## Results

| Scene | Inlier Rate | Mean Reprojection Error |
|-------|-------------|------------------------|
| v_bird | > 80% | < 2 px |
| v_boat | > 80% | < 2 px |
| v_circus | > 80% | < 2 px |
| v_graffiti | > 80% | < 2 px |
| v_soldiers | > 80% | < 2 px |
| v_weapons | > 80% | < 2 px |

---

## Dependencies

- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Pandas

See [requirements.txt](requirements.txt) for pinned versions.

---

## License

MIT
