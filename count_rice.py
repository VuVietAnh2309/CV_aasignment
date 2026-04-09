import cv2
import numpy as np
import sys
import os
from scipy.signal import medfilt


def remove_sinusoidal_noise(gray):
    """Detect and remove sinusoidal noise by subtracting column-mean pattern."""
    gray_f = gray.astype(np.float64)
    col_means = np.mean(gray_f, axis=0)
    baseline = medfilt(col_means, kernel_size=101)
    sinus_1d = col_means - baseline

    # Check if there is significant sinusoidal component
    energy = np.std(sinus_1d)
    if energy > 3:  # significant periodic noise detected
        sinus_2d = np.tile(sinus_1d, (gray.shape[0], 1))
        corrected = np.clip(gray_f - sinus_2d, 0, 255).astype(np.uint8)
        return corrected
    return gray


def count_rice(image_path):
    # 1. Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return -1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Remove sinusoidal noise (on raw grayscale)
    denoised = remove_sinusoidal_noise(gray)

    # 3. Denoise: median (salt&pepper) + Gaussian (smooth)
    denoised = cv2.medianBlur(denoised, 5)
    denoised = cv2.GaussianBlur(denoised, (5, 5), 1)

    # 4. Segmentation: adaptive threshold
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, -8
    )

    # Ensure rice grains are white
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # 5. Morphological post-processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 6. Watershed to separate touching grains
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_ws = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_ws, markers)

    # 7. Count objects filtered by area
    result_mask = np.zeros_like(binary)
    result_mask[markers > 1] = 255

    contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    if len(areas) == 0:
        return 0

    median_area = np.median(areas)
    min_area = median_area * 0.15
    max_area = median_area * 4.0

    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            count += 1
        elif area > max_area:
            count += round(area / median_area)

    return count


if __name__ == "__main__":
    if len(sys.argv) < 2:
        image_dir = os.path.join(os.path.dirname(__file__), "Proj1.2")
        for fname in sorted(os.listdir(image_dir)):
            if fname.endswith(".png") or fname.endswith(".jpg"):
                path = os.path.join(image_dir, fname)
                n = count_rice(path)
                print(f"{fname}: {n} grains")
    else:
        path = sys.argv[1]
        n = count_rice(path)
        print(f"Number of rice grains: {n}")
