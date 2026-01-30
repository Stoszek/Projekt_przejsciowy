import os
import csv
import math
import random
import time
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from numba import jit, prange
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


# ====================== 1. Generowanie jąder ======================

def generate_mean_kernel(size):
    if size % 2 == 0:
        raise ValueError("Rozmiar jądra musi być nieparzysty.")
    return np.full((size, size), 1 / (size * size), dtype=np.float32)


def generate_gaussian_kernel(size, sigma):
    if size % 2 == 0:
        raise ValueError("Rozmiar jądra musi być nieparzysty.")
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    kernel /= (2 * math.pi * sigma ** 2)
    kernel /= kernel.sum()
    return kernel


# ====================== 2. Nakładanie szumu (zoptymalizowane) ======================

def add_noise(image, noise_type, intensity=0.1, seed=None, region=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if noise_type not in ['gaussian', 'salt_pepper', 'speckle']:
        raise ValueError("Nieobsługiwany typ szumu.")

    noisy_image = image.copy()
    h, w = image.shape[:2]

    quarters = [
        (0, h // 2, 0, w // 2),
        (0, h // 2, w // 2, w),
        (h // 2, h, 0, w // 2),
        (h // 2, h, w // 2, w),
    ]

    regions_to_apply = region if region else [True] * 4

    for i, apply_noise in enumerate(regions_to_apply):
        if not apply_noise:
            continue

        y1, y2, x1, x2 = quarters[i]
        local_region = noisy_image[y1:y2, x1:x2]

        if noise_type == 'gaussian':
            noisy_region = apply_gaussian_noise(local_region, intensity)
        elif noise_type == 'salt_pepper':
            noisy_region = apply_salt_pepper_noise(local_region, intensity)
        elif noise_type == 'speckle':
            noisy_region = apply_speckle_noise(local_region, intensity)

        noisy_image[y1:y2, x1:x2] = noisy_region

    return noisy_image


def apply_gaussian_noise(region, intensity):
    sigma = intensity * 255
    gauss = np.random.normal(0, sigma, region.shape).astype(np.float32)
    noisy_region = region.astype(np.float32) + gauss
    mean_original = np.mean(region)
    mean_noisy = np.mean(noisy_region)
    noisy_region += (mean_original - mean_noisy)
    return np.clip(noisy_region, 0, 255).astype(np.uint8)


def apply_salt_pepper_noise(region, intensity):
    prob = intensity
    noisy_region = region.copy()
    h, w = region.shape[:2]

    # Generuj losowe wartości dla całego regionu naraz
    rand_vals = np.random.random((h, w))

    # Zastosuj szum salt & pepper wektorowo
    salt_mask = rand_vals < prob / 2
    pepper_mask = (rand_vals >= prob / 2) & (rand_vals < prob)

    noisy_region[salt_mask] = 0
    noisy_region[pepper_mask] = 255

    return noisy_region


def apply_speckle_noise(region, intensity):
    gauss = np.random.randn(*region.shape) * intensity
    noisy_region = region.astype(np.float32) * (1 + gauss)
    mean_original = np.mean(region)
    mean_noisy = np.mean(noisy_region)
    noisy_region += (mean_original - mean_noisy)
    return np.clip(noisy_region, 0, 255).astype(np.uint8)


# ====================== 3. Filtry podstawowe (zoptymalizowane z Numba) ======================

@jit(nopython=True, parallel=True, cache=True)
def apply_filter_numba(image_array, kernel, width, height, channels):
    """Szybka konwolucja z użyciem Numba"""
    size = kernel.shape[0]
    offset = size // 2
    output = np.zeros((height, width, channels), dtype=np.float32)

    for c in prange(channels):
        for y in range(height):
            for x in range(width):
                acc = 0.0
                for i in range(size):
                    for j in range(size):
                        xi = x + i - offset
                        yj = y + j - offset
                        if 0 <= xi < width and 0 <= yj < height:
                            acc += image_array[yj, xi, c] * kernel[i, j]
                output[y, x, c] = acc

    return np.clip(output, 0, 255).astype(np.uint8)


def apply_filter(image, kernel):
    """Filtr uśredniający (mean)"""
    image_array = np.array(image, dtype=np.float32)
    kernel_array = np.array(kernel, dtype=np.float32)
    height, width, channels = image_array.shape

    filtered = apply_filter_numba(image_array, kernel_array, width, height, channels)
    return Image.fromarray(filtered)


def apply_gaussian_filter(image, kernel):
    """Filtr Gaussa"""
    return apply_filter(image, kernel)


# ====================== 4. Filtr bilateralny (zoptymalizowany) ======================

@jit(nopython=True, cache=True)
def gaussian_jit(x, sigma):
    return math.exp(-(x ** 2) / (2 * sigma ** 2))


@jit(nopython=True, parallel=True, cache=True)
def apply_bilateral_filter_numba(image_array, spatial_sigma, intensity_sigma, kernel_size, offset):
    height, width, channels = image_array.shape
    output_image = np.zeros((height, width, channels), dtype=np.float32)

    # Pre-compute spatial kernel
    spatial_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            dist = math.sqrt((i - offset) ** 2 + (j - offset) ** 2)
            spatial_kernel[i, j] = gaussian_jit(dist, spatial_sigma)

    for y in prange(height):
        for x in range(width):
            for c in range(channels):
                center_intensity = image_array[y, x, c]
                weighted_sum = 0.0
                weight_sum = 0.0

                for i in range(kernel_size):
                    for j in range(kernel_size):
                        ny = y + i - offset
                        nx = x + j - offset

                        if 0 <= ny < height and 0 <= nx < width:
                            neighbor_intensity = image_array[ny, nx, c]
                            intensity_diff = neighbor_intensity - center_intensity
                            intensity_weight = gaussian_jit(intensity_diff, intensity_sigma)
                            weight = spatial_kernel[i, j] * intensity_weight

                            weighted_sum += neighbor_intensity * weight
                            weight_sum += weight

                output_image[y, x, c] = weighted_sum / weight_sum if weight_sum > 0 else center_intensity

    return np.clip(output_image, 0, 255).astype(np.uint8)


def apply_bilateral_filter(image, spatial_sigma, intensity_sigma):
    image_array = np.array(image, dtype=np.float32)
    kernel_size = int(6 * spatial_sigma) | 1
    offset = kernel_size // 2

    filtered = apply_bilateral_filter_numba(image_array, spatial_sigma, intensity_sigma, kernel_size, offset)
    return Image.fromarray(filtered)


# ====================== 5. Filtr adaptacyjny (zoptymalizowany) ======================

@jit(nopython=True, cache=True)
def median_filter_custom_numba(image, size):
    h, w = image.shape
    pad_size = size // 2
    filtered_image = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            # Zbierz wartości z sąsiedztwa
            values = []
            for di in range(-pad_size, pad_size + 1):
                for dj in range(-pad_size, pad_size + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        values.append(image[ni, nj])

            # Sortuj i znajdź medianę
            values_array = np.array(values)
            values_array.sort()
            filtered_image[i, j] = values_array[len(values_array) // 2]

    return filtered_image


@jit(nopython=True, cache=True)
def convolve_numba(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(image_height):
        for j in range(image_width):
            acc = 0.0
            for ki in range(kernel_height):
                for kj in range(kernel_width):
                    ni = i + ki - pad_height
                    nj = j + kj - pad_width
                    if 0 <= ni < image_height and 0 <= nj < image_width:
                        acc += image[ni, nj] * kernel[ki, kj]
            result[i, j] = acc

    return result


def compute_local_stats(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    local_mean = convolve_numba(image, kernel)
    squared_diff = (image - local_mean) ** 2
    local_variance = convolve_numba(squared_diff, kernel)
    return local_mean, local_variance


def estimate_noise_variance(image, kernel_size):
    smoothed_image = median_filter_custom_numba(image, size=kernel_size)
    noise = image - smoothed_image
    return np.var(noise)


def apply_adaptive_filter(image, noise_variance, kernel_size):
    local_mean, local_variance = compute_local_stats(image, kernel_size)
    filtered_image = image - (noise_variance / np.maximum(local_variance, noise_variance)) * (image - local_mean)
    return np.clip(filtered_image, 0, 255).astype(np.uint8)


# ====================== 6. Metryki jakości ======================

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    ssim_value, _ = ssim(img1, img2, full=True, channel_axis=-1, win_size=3)
    return ssim_value


def calculate_ncd(img1, img2):
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)
    delta_E = np.linalg.norm(img1_lab.astype(np.float32) - img2_lab.astype(np.float32), axis=2)
    max_delta_e = math.sqrt(100 ** 2 + 255 ** 2 + 255 ** 2)
    return np.mean(delta_E / max_delta_e)


# ====================== 7. Pomocnicze funkcje ======================

def save_image(image, filename):
    if isinstance(image, np.ndarray):
        Image.fromarray(image).save(filename)
    elif isinstance(image, Image.Image):
        image.save(filename)


def generate_filename(base_name, operation, params):
    params_str = "_".join([f"{key}{value}" for key, value in params.items()])
    return f"{base_name}_{operation}_{params_str}.png"


def save_results_to_csv(results, filename="results.csv"):
    headers = [
        "Filter", "Noise Type", "Intensity", "Kernel Size", "Sigma",
        "Spatial Sigma", "Intensity Sigma", "Median Kernel Size",
        "PSNR", "SSIM", "NCD", "File", "Execution Time (s)"
    ]
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)


# ====================== 8. Funkcje pomocnicze do przetwarzania równoległego ======================

def process_mean_filter(args):
    """Przetwarza jeden przypadek filtra mean"""
    noisy_image, original_image, base_name, noise_type, intensity, kernel_size = args

    start_time = time.time()
    kernel = generate_mean_kernel(kernel_size)
    filtered_pil = apply_filter(Image.fromarray(noisy_image), kernel)
    execution_time = time.time() - start_time

    filtered_filename = generate_filename(base_name, "filtered_mean",
                                          {"noise_type": noise_type, "intensity": intensity,
                                           "kernel_size": kernel_size})
    save_image(filtered_pil, filtered_filename)
    filtered_np = np.array(filtered_pil)

    return {
        "Filter": "Mean", "Noise Type": noise_type, "Intensity": intensity,
        "Kernel Size": kernel_size, "Sigma": None, "Spatial Sigma": None,
        "Intensity Sigma": None, "Median Kernel Size": None,
        "PSNR": calculate_psnr(original_image, filtered_np),
        "SSIM": calculate_ssim(original_image, filtered_np),
        "NCD": calculate_ncd(original_image, filtered_np),
        "Execution Time (s)": round(execution_time, 4),
        "File": filtered_filename
    }


def process_gaussian_filter(args):
    """Przetwarza jeden przypadek filtra Gaussa"""
    noisy_image, original_image, base_name, noise_type, intensity, kernel_size, sigma = args

    start_time = time.time()
    kernel = generate_gaussian_kernel(kernel_size, sigma)
    filtered_pil = apply_gaussian_filter(Image.fromarray(noisy_image), kernel)
    execution_time = time.time() - start_time

    filtered_filename = generate_filename(base_name, "filtered_gauss",
                                          {"noise_type": noise_type, "intensity": intensity,
                                           "kernel_size": kernel_size, "sigma": sigma})
    save_image(filtered_pil, filtered_filename)
    filtered_np = np.array(filtered_pil)

    return {
        "Filter": "Gaussian", "Noise Type": noise_type, "Intensity": intensity,
        "Kernel Size": kernel_size, "Sigma": sigma, "Spatial Sigma": None,
        "Intensity Sigma": None, "Median Kernel Size": None,
        "PSNR": calculate_psnr(original_image, filtered_np),
        "SSIM": calculate_ssim(original_image, filtered_np),
        "NCD": calculate_ncd(original_image, filtered_np),
        "Execution Time (s)": round(execution_time, 4),
        "File": filtered_filename
    }


def process_bilateral_filter(args):
    """Przetwarza jeden przypadek filtra bilateralnego"""
    noisy_image, original_image, base_name, noise_type, intensity, spatial_sigma, intensity_sigma = args

    start_time = time.time()
    filtered_pil = apply_bilateral_filter(Image.fromarray(noisy_image.astype(np.uint8)),
                                          spatial_sigma, intensity_sigma)
    execution_time = time.time() - start_time

    filtered_filename = generate_filename(base_name, "filtered_bilateral",
                                          {"noise_type": noise_type, "intensity": intensity,
                                           "spatial_sigma": spatial_sigma, "intensity_sigma": intensity_sigma})
    save_image(filtered_pil, filtered_filename)
    filtered_np = np.array(filtered_pil)

    return {
        "Filter": "Bilateral", "Noise Type": noise_type, "Intensity": intensity,
        "Kernel Size": None, "Sigma": None, "Spatial Sigma": spatial_sigma,
        "Intensity Sigma": intensity_sigma, "Median Kernel Size": None,
        "PSNR": calculate_psnr(original_image, filtered_np),
        "SSIM": calculate_ssim(original_image, filtered_np),
        "NCD": calculate_ncd(original_image, filtered_np),
        "Execution Time (s)": round(execution_time, 4),
        "File": filtered_filename
    }


def process_adaptive_filter(args):
    """Przetwarza jeden przypadek filtra adaptacyjnego"""
    noisy_image, original_image, base_name, noise_type, intensity, kernel_size, median_kernel_size = args

    start_time = time.time()
    filtered_channels = []
    for channel_idx in range(3):
        channel_array = noisy_image[:, :, channel_idx].astype(np.float32)
        noise_variance = estimate_noise_variance(channel_array, kernel_size=median_kernel_size)
        filtered_channel = apply_adaptive_filter(channel_array, noise_variance, kernel_size=kernel_size)
        filtered_channels.append(filtered_channel)
    filtered_image_array = np.stack(filtered_channels, axis=2).astype(np.uint8)
    execution_time = time.time() - start_time

    filtered_filename = generate_filename(base_name, "filtered_adaptive",
                                          {"noise_type": noise_type, "intensity": intensity,
                                           "median_kernel_size": median_kernel_size, "kernel_size": kernel_size})
    save_image(filtered_image_array, filtered_filename)

    return {
        "Filter": "Adaptive", "Noise Type": noise_type, "Intensity": intensity,
        "Kernel Size": kernel_size, "Sigma": None, "Spatial Sigma": None,
        "Intensity Sigma": None, "Median Kernel Size": median_kernel_size,
        "PSNR": calculate_psnr(original_image, filtered_image_array),
        "SSIM": calculate_ssim(original_image, filtered_image_array),
        "NCD": calculate_ncd(original_image, filtered_image_array),
        "Execution Time (s)": round(execution_time, 4),
        "File": filtered_filename
    }


# ====================== 9. Główna automatyzacja z równoległym przetwarzaniem ======================

def main_automatyzacja():
    input_image_name = "kodim04.png"
    if not os.path.exists(input_image_name):
        print(f"Plik {input_image_name} nie istnieje!")
        return

    base_name = os.path.splitext(input_image_name)[0]
    original_image = np.array(Image.open(input_image_name))
    seed = 42

    noise_types = ['gaussian', 'salt_pepper', 'speckle']
    noise_intensities = [0.1, 0.3, 0.5]
    kernel_sizes = [3, 5, 7]
    sigmas = [1.0, 10.0, 25.0]
    spatial_sigmas = [5, 9, 13]
    intensity_sigmas = [50, 90, 130]

    results = []

    # Użyj ProcessPoolExecutor do przetwarzania równoległego
    max_workers = os.cpu_count()

    print(f"Rozpoczynam przetwarzanie z {max_workers} procesami równoległymi...")

    for noise_type in noise_types:
        for intensity in noise_intensities:
            print(f"Przetwarzanie: {noise_type} noise, intensity={intensity}")

            noisy_image = add_noise(original_image, noise_type, intensity, seed)
            noisy_filename = generate_filename(base_name, "noisy", {"type": noise_type, "intensity": intensity})
            save_image(noisy_image, noisy_filename)

            # Przygotuj zadania dla wszystkich filtrów
            tasks = []

            # Mean filter tasks
            for kernel_size in kernel_sizes:
                tasks.append(('mean', (noisy_image, original_image, base_name, noise_type, intensity, kernel_size)))

            # Gaussian filter tasks
            for kernel_size in kernel_sizes:
                for sigma in sigmas:
                    tasks.append(('gaussian',
                                  (noisy_image, original_image, base_name, noise_type, intensity, kernel_size, sigma)))

            # Bilateral filter tasks
            for spatial_sigma in spatial_sigmas:
                for intensity_sigma in intensity_sigmas:
                    tasks.append(('bilateral',
                                  (noisy_image, original_image, base_name, noise_type, intensity, spatial_sigma,
                                   intensity_sigma)))

            # Adaptive filter tasks
            for kernel_size in kernel_sizes:
                for median_kernel_size in kernel_sizes:
                    tasks.append(('adaptive',
                                  (noisy_image, original_image, base_name, noise_type, intensity, kernel_size,
                                   median_kernel_size)))

            # Wykonaj zadania równolegle
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for filter_type, args in tasks:
                    if filter_type == 'mean':
                        futures.append(executor.submit(process_mean_filter, args))
                    elif filter_type == 'gaussian':
                        futures.append(executor.submit(process_gaussian_filter, args))
                    elif filter_type == 'bilateral':
                        futures.append(executor.submit(process_bilateral_filter, args))
                    elif filter_type == 'adaptive':
                        futures.append(executor.submit(process_adaptive_filter, args))

                # Zbierz wyniki
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Błąd podczas przetwarzania: {e}")

    save_results_to_csv(results)
    print("Automatyzacja zakończona. Wyniki zapisano do results.csv oraz wygenerowano wszystkie obrazy.")


# ====================== Uruchomienie ======================

if __name__ == "__main__":
    main_automatyzacja()