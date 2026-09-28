from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2 as cv
import numpy as np


def resize_for_work(img: np.ndarray, target_mp: float = 0.8) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, math.sqrt(target_mp * 1_000_000 / max(1, h * w)))
    if scale >= 0.999:
        return img
    return cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)


def fill_poles(canvas: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = canvas.copy()
    valid = mask > 0

    # Fill only missing vertical polar regions from the nearest valid pixel in each column.
    for x in range(w):
        ys = np.flatnonzero(valid[:, x])
        if ys.size == 0:
            continue
        top, bot = int(ys[0]), int(ys[-1])
        if top > 0:
            out[:top, x] = out[top, x]
        if bot < h - 1:
            out[bot + 1 :, x] = out[bot, x]

    # Fill small internal holes without letting them dominate the scene.
    remaining = (mask == 0).astype(np.uint8) * 255
    if np.count_nonzero(remaining) > 0:
        small = cv.resize(remaining, (max(32, w // 4), max(16, h // 4)), interpolation=cv.INTER_NEAREST)
        small_out = cv.resize(out, (small.shape[1], small.shape[0]), interpolation=cv.INTER_AREA)
        try:
            small_out = cv.inpaint(small_out, small, 3, cv.INPAINT_TELEA)
            repaired = cv.resize(small_out, (w, h), interpolation=cv.INTER_CUBIC)
            out[mask == 0] = repaired[mask == 0]
        except cv.error:
            pass
    return out


def place_wrapped(dst: np.ndarray, dst_mask: np.ndarray, src: np.ndarray, src_mask: np.ndarray, x0: int, y0: int):
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    y1 = max(0, y0)
    y2 = min(H, y0 + h)
    if y2 <= y1:
        return
    sy1, sy2 = y1 - y0, y2 - y0

    for offset in (0, -W, W):
        xx = x0 + offset
        x1 = max(0, xx)
        x2 = min(W, xx + w)
        if x2 <= x1:
            continue
        sx1, sx2 = x1 - xx, x2 - xx
        m = src_mask[sy1:sy2, sx1:sx2] > 0
        target = dst[y1:y2, x1:x2]
        target_mask = dst_mask[y1:y2, x1:x2]
        patch = src[sy1:sy2, sx1:sx2]
        target[m] = patch[m]
        target_mask[m] = 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--qa", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bundle-adjust", action="store_true")
    mode.add_argument("--no-bundle-adjust", action="store_true")
    args = ap.parse_args()

    started = time.time()
    names = [Path(args.input) / f"{i:02d}.jpg" for i in range(1, 37)]
    originals = [cv.imread(str(p), cv.IMREAD_COLOR) for p in names]
    if any(im is None for im in originals):
        raise RuntimeError("One or more capture images could not be decoded")

    images = [resize_for_work(im, 0.8) for im in originals]
    finder = cv.SIFT_create(nfeatures=3500, contrastThreshold=0.025)
    features = [cv.detail.computeImageFeatures2(finder, im) for im in images]

    # 13 is deliberate: it links neighbors in each 12-image ring and corresponding
    # directions between adjacent rings, without doing every possible pair.
    matcher = cv.detail_BestOf2NearestRangeMatcher(13, False, 0.45)
    pairwise_matches = matcher.apply2(features)
    matcher.collectGarbage()

    component = cv.detail.leaveBiggestComponent(features, pairwise_matches, 0.35)
    kept = [int(i) for i in component]
    if len(kept) < 24:
        raise RuntimeError(f"Only {len(kept)}/36 images formed a connected panorama")

    images = [images[i] for i in kept]
    names = [names[i] for i in kept]

    estimator = cv.detail_HomographyBasedEstimator()
    ok, cameras = estimator.apply(features, pairwise_matches, None)
    if not ok:
        raise RuntimeError("Homography camera estimation failed")
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    if args.bundle_adjust:
        adjuster = cv.detail_BundleAdjusterRay()
        adjuster.setConfThresh(0.35)
        refine = np.zeros((3, 3), np.uint8)
        refine[0, 0] = 1
        refine[0, 1] = 1
        refine[0, 2] = 1
        refine[1, 1] = 1
        refine[1, 2] = 1
        adjuster.setRefinementMask(refine)
        ok, cameras = adjuster.apply(features, pairwise_matches, cameras)
        if not ok:
            raise RuntimeError("Bundle adjustment failed")

    rmats = [np.copy(cam.R) for cam in cameras]
    rmats = cv.detail.waveCorrect(rmats, cv.detail.WAVE_CORRECT_HORIZ)
    for i, cam in enumerate(cameras):
        cam.R = rmats[i]

    focals = sorted(float(cam.focal) for cam in cameras if np.isfinite(cam.focal) and cam.focal > 1)
    if not focals:
        raise RuntimeError("No valid focal lengths")
    scale = focals[len(focals) // 2]

    warper = cv.PyRotationWarper("spherical", scale)
    corners, warped, masks, sizes = [], [], [], []
    for img, cam in zip(images, cameras):
        K = cam.K().astype(np.float32)
        corner, wi = warper.warp(img, K, cam.R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        mask = np.full(img.shape[:2], 255, np.uint8)
        _, wm = warper.warp(mask, K, cam.R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        corners.append(corner)
        warped.append(wi)
        masks.append(wm)
        sizes.append((wi.shape[1], wi.shape[0]))

    compensator = cv.detail.ExposureCompensator_createDefault(cv.detail.ExposureCompensator_GAIN_BLOCKS)
    compensator.feed(corners, warped, masks)

    try:
        seam_finder = cv.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
        masks = seam_finder.find([w.astype(np.float32) for w in warped], corners, masks)
        seam_method = "graphcut-colorgrad"
    except cv.error:
        seam_finder = cv.detail_DpSeamFinder("COLOR_GRAD")
        masks = seam_finder.find([w.astype(np.float32) for w in warped], corners, masks)
        seam_method = "dp-colorgrad"

    roi = cv.detail.resultRoi(corners=corners, sizes=sizes)
    blend_width = math.sqrt(max(1, roi[2] * roi[3])) * 0.05
    blender = cv.detail_MultiBandBlender()
    blender.setNumBands(max(3, min(7, int(math.log(max(2.0, blend_width), 2) - 1))))
    blender.prepare(roi)

    for i, (wi, wm, corner) in enumerate(zip(warped, masks, corners)):
        compensator.apply(i, corner, wi, wm)
        blender.feed(cv.UMat(wi.astype(np.int16)), wm, corner)

    result, result_mask = blender.blend(None, None)
    result = np.clip(result, 0, 255).astype(np.uint8)
    result_mask = result_mask.get() if isinstance(result_mask, cv.UMat) else result_mask

    # Spherical warper coordinates are approximately x=scale*longitude,
    # y=scale*latitude. Reconstruct a fixed full equirectangular canvas.
    sphere_w = max(1024, int(round(2 * math.pi * scale)))
    sphere_h = max(512, int(round(math.pi * scale)))
    full = np.zeros((sphere_h, sphere_w, 3), np.uint8)
    full_mask = np.zeros((sphere_h, sphere_w), np.uint8)
    x0 = int(round(roi[0] + math.pi * scale))
    y0 = int(round(roi[1] + 0.5 * math.pi * scale))
    place_wrapped(full, full_mask, result, result_mask, x0, y0)

    coverage = float(np.count_nonzero(full_mask)) / float(full_mask.size)
    if coverage < 0.35:
        raise RuntimeError(f"Spherical coverage too low: {coverage:.3f}")

    full = fill_poles(full, full_mask)
    final = cv.resize(full, (4096, 2048), interpolation=cv.INTER_LANCZOS4)
    if not cv.imwrite(args.output, final, [cv.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError("Could not save panorama")

    qa = {
        "version": "v2.0",
        "input_images": 36,
        "connected_images": len(kept),
        "bundle_adjustment": bool(args.bundle_adjust),
        "seam": seam_method,
        "coverage_before_pole_fill": round(coverage, 4),
        "warped_scale": round(float(scale), 3),
        "output_width": 4096,
        "output_height": 2048,
        "seconds": round(time.time() - started, 2),
    }
    Path(args.qa).write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
