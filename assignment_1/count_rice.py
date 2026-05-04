"""
Rice Grain Counter - Image Processing Pipeline

Pipeline:
  1. Grayscale conversion
  2. Periodic/sinusoidal noise detection & removal
     - Primary: column-mean subtraction (fast, conservative)
     - Alternative: 2D Fourier notch filter (provided, demonstration)
  3. Median filter (salt & pepper noise)
  4. Gaussian blur (smoothing)
  5. CLAHE (local contrast enhancement, critical for uneven illumination)
  6. Adaptive Threshold (robust to uneven illumination, unlike Otsu)
  7. Morphological Opening + Closing
  8. Watershed (separate touching grains)
  9. Smart area-based filtering (median-relative + cluster estimation)

Usage:
  python count_rice.py <image_path>           # Count one image
  python count_rice.py                         # Count all images in Proj1.2/
  python count_rice.py <image_path> --debug    # Save intermediate steps
"""

import cv2
import numpy as np
import sys
import os
from scipy.signal import medfilt


# ============================================================
# Step 2a: Column-mean subtraction (primary method)
# ============================================================

def remove_sinusoidal_noise_columnwise(gray, energy_threshold=3.0,
                                       median_kernel=101):
    """
    Detect and remove vertical sinusoidal stripes via column-mean subtraction.

    For each column, compute mean intensity across rows -> get a 1D profile.
    Baseline (low-frequency) is extracted with a wide median filter.
    The residual (profile - baseline) is the sinusoidal pattern.

    Only applied if the residual's energy (std) exceeds threshold - avoids
    destroying clean images.

    Args:
        gray: grayscale image (uint8)
        energy_threshold: minimum std of residual to trigger correction
        median_kernel: kernel size for baseline extraction (must be odd)

    Returns:
        corrected grayscale image (uint8), bool (True if correction applied)
    """
    gray_f = gray.astype(np.float64)
    col_means = np.mean(gray_f, axis=0)
    baseline = medfilt(col_means, kernel_size=median_kernel)
    residual = col_means - baseline

    if np.std(residual) <= energy_threshold:
        return gray, False

    sinus_2d = np.tile(residual, (gray.shape[0], 1))
    corrected = np.clip(gray_f - sinus_2d, 0, 255).astype(np.uint8)
    return corrected, True


# ============================================================
# Step 2b: 2D Fourier notch filter (alternative method)
# ============================================================

def remove_periodic_noise_fourier(gray, dc_radius=30, notch_radius=5,
                                  peak_ratio=0.5, max_peaks=8):
    """
    Remove periodic noise using 2D Fourier notch filter.

    More general than column-mean: handles periodic noise in any direction.
    Provided for demonstration - NOT used by default because automatic peak
    detection on rice images is unreliable (rice texture also creates peaks).

    Algorithm:
      1. Compute FFT and shift DC to center.
      2. Mask out DC region (low frequencies = image content).
      3. Find top-K peaks in log-magnitude spectrum.
      4. Apply notch filter at each peak (zero out small circles).
      5. Inverse FFT.

    Args:
        gray: grayscale image (uint8)
        dc_radius: radius around DC to exclude from peak search
        notch_radius: radius of notch around each peak
        peak_ratio: peak must be >= peak_ratio * max (outside DC) to be kept
        max_peaks: hard cap on number of notches applied (conservative)

    Returns:
        filtered grayscale image (uint8)
    """
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    # Exclude DC region
    yy, xx = np.ogrid[:rows, :cols]
    dc_mask = (xx - ccol) ** 2 + (yy - crow) ** 2 <= dc_radius ** 2
    search = mag.copy()
    search[dc_mask] = 0

    max_val = search.max()
    if max_val < 1e-6:
        return gray

    threshold = peak_ratio * max_val
    # Get candidate peaks, sort by magnitude, take top-K
    candidates = np.argwhere(search >= threshold)
    if len(candidates) == 0:
        return gray
    vals = search[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(-vals)[:max_peaks]
    peaks = candidates[order]

    # Build notch filter
    notch = np.ones((rows, cols), dtype=np.float64)
    for py, px_ in peaks:
        circle = (xx - px_) ** 2 + (yy - py) ** 2 <= notch_radius ** 2
        notch[circle] = 0

    fshift_filtered = fshift * notch
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    return np.clip(img_back, 0, 255).astype(np.uint8)


# ============================================================
# Step 5: CLAHE
# ============================================================

def apply_clahe(gray, clip_limit=2.8, tile_grid_size=(7, 7)):
    """
    Contrast Limited Adaptive Histogram Equalization.

    Divides image into tiles, equalizes histogram locally, clips to avoid
    amplifying noise. Critical for uneven illumination - enhances local
    contrast so rice grains stand out from the background everywhere.

    Args:
        gray: grayscale image (uint8)
        clip_limit: histogram clip threshold (higher = stronger contrast)
        tile_grid_size: (rows, cols) of tiles

    Returns:
        contrast-enhanced grayscale image (uint8)
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def needs_contrast_enhancement(gray, mean_threshold=110, std_threshold=40):
    """
    Decide whether CLAHE should be applied, based on image statistics.

    CLAHE helps images with low brightness OR low contrast - but on clean
    bright images it amplifies residual noise into false positives.
    So we only apply it when the image actually needs it.

    Returns:
        True if mean brightness is low OR standard deviation is low.
    """
    return gray.mean() < mean_threshold or gray.std() < std_threshold


# ============================================================
# Main pipeline
# ============================================================

def count_rice(image_path, debug_dir=None):
    """
    Count rice grains in an image.

    Args:
        image_path: path to input image
        debug_dir: if set, save intermediate images here (for report)

    Returns:
        number of rice grains detected (int), or -1 on error
    """
    # 1. Read and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return -1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _save_debug(debug_dir, "01_gray", gray)

    # Decide CLAHE need from ORIGINAL image stats, before smoothing
    # (median + Gaussian reduce std and mislead the detector).
    use_clahe = needs_contrast_enhancement(gray)

    # 2. Remove sinusoidal noise (conditional, column-mean based)
    denoised, applied = remove_sinusoidal_noise_columnwise(gray)
    _save_debug(debug_dir, "02_sinusoidal_removed", denoised)

    # 3. Median filter - removes salt & pepper while preserving edges
    denoised = cv2.medianBlur(denoised, 5)
    _save_debug(debug_dir, "03_median", denoised)

    # 4. Gaussian blur - smooth out fine-grained noise
    denoised = cv2.GaussianBlur(denoised, (5, 5), 1)
    _save_debug(debug_dir, "04_gaussian", denoised)

    # 5. CLAHE - enhance local contrast ONLY if image needs it
    # Applied for dark / low-contrast images (uneven illumination).
    # Skipped for well-exposed images to avoid amplifying residual noise
    # (decided from the ORIGINAL image stats in step 1).
    if use_clahe:
        enhanced = apply_clahe(denoised, clip_limit=2.8, tile_grid_size=(7, 7))
    else:
        enhanced = denoised
    _save_debug(debug_dir, "05_clahe", enhanced)

    # 6. Adaptive threshold - robust to uneven illumination (unlike Otsu)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, -8
    )
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    _save_debug(debug_dir, "06_threshold", binary)

    # 7. Morphological post-processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    _save_debug(debug_dir, "07_morphology", binary)

    # 8. Watershed - separate touching grains
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_ws = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_ws, markers)
    _save_debug(debug_dir, "08_watershed", _visualize_watershed(markers))

    # 9. Extract contours and filter by area (median-relative)
    result_mask = np.zeros_like(binary)
    result_mask[markers > 1] = 255

    contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if len(areas) == 0:
        return 0

    median_area = np.median(areas)
    min_area = median_area * 0.15
    max_area = median_area * 4.0

    count = 0
    kept_contours = []
    for c, area in zip(contours, areas):
        if min_area <= area <= max_area:
            count += 1
            kept_contours.append((c, 1))
        elif area > max_area:
            n_in_cluster = round(area / median_area)
            count += n_in_cluster
            kept_contours.append((c, n_in_cluster))

    # Save final visualization with bounding boxes
    if debug_dir is not None:
        vis = img.copy()
        for c, n in kept_contours:
            x, y, w, h = cv2.boundingRect(c)
            color = (0, 255, 0) if n == 1 else (0, 165, 255)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            if n > 1:
                cv2.putText(vis, f"x{n}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(vis, f"Total: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        _save_debug(debug_dir, "09_final", vis)

    return count


# ============================================================
# Helpers
# ============================================================

def _save_debug(debug_dir, name, img):
    """Save an intermediate image if debug mode is on."""
    if debug_dir is None:
        return
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), img)


def _visualize_watershed(markers):
    """Convert watershed markers to a visualization image."""
    vis = np.zeros((*markers.shape, 3), dtype=np.uint8)
    num_labels = markers.max()
    if num_labels > 1:
        colors = np.random.RandomState(42).randint(50, 255, (num_labels + 2, 3))
        for label in range(2, num_labels + 1):
            vis[markers == label] = colors[label]
    vis[markers == -1] = (0, 0, 255)
    return vis


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    args = sys.argv[1:]
    debug = "--debug" in args
    args = [a for a in args if a != "--debug"]

    if len(args) == 0:
        image_dir = os.path.join(os.path.dirname(__file__), "Proj1.2")
        print(f"{'Image':<60} {'Count':>6}")
        print("-" * 70)
        for fname in sorted(os.listdir(image_dir)):
            if fname.endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(image_dir, fname)
                if debug:
                    dbg = os.path.join(os.path.dirname(__file__), "debug",
                                       os.path.splitext(fname)[0])
                    n = count_rice(path, debug_dir=dbg)
                else:
                    n = count_rice(path)
                print(f"{fname:<60} {n:>6}")
    else:
        path = args[0]
        if debug:
            name = os.path.splitext(os.path.basename(path))[0]
            dbg = os.path.join(os.path.dirname(path) or ".", "debug", name)
            n = count_rice(path, debug_dir=dbg)
            print(f"Debug images saved to: {dbg}")
        else:
            n = count_rice(path)
        print(f"Number of rice grains: {n}")
