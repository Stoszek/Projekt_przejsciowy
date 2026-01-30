import cv2
import numpy as np
import random


def add_noise(image, noise_type, intensity=0.1, seed=None, region=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if noise_type not in ['gaussian', 'salt_pepper', 'speckle']:
        raise ValueError("Nieobsługiwany typ szumu. Wybierz 'gaussian', 'salt_pepper' lub 'speckle'.")

    noisy_image = image.copy()
    h, w, _ = image.shape

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
    gauss = np.random.normal(0, sigma, region.shape).astype('float32')
    noisy_region = region.astype('float32') + gauss
    mean_original = np.mean(region)
    mean_noisy = np.mean(noisy_region)
    noisy_region += (mean_original - mean_noisy)
    return np.clip(noisy_region, 0, 255).astype('uint8')

def apply_salt_pepper_noise(region, intensity):
    prob = intensity
    noisy_region = region.copy()
    for i in range(region.shape[0]):
        for j in range(region.shape[1]):
            rand = random.random()
            if rand < prob / 2:
                noisy_region[i, j] = 0
            elif rand < prob:
                noisy_region[i, j] = 255
    return noisy_region

def apply_speckle_noise(region, intensity):
    gauss = np.random.randn(*region.shape) * intensity
    noisy_region = region.astype('float32') + region.astype('float32') * gauss
    mean_original = np.mean(region)
    mean_noisy = np.mean(noisy_region)
    noisy_region += (mean_original - mean_noisy)

    return np.clip(noisy_region, 0, 255).astype('uint8')

if __name__ == "__main__":
    try:
        image = cv2.imread("kodim04.png")
        if image is None:
            raise FileNotFoundError("Nie znaleziono pliku. Upewnij się, że plik istnieje.")

        cv2.imshow("Original", image)
        intensity = 0.1
        gaussian_noise = add_noise(image, noise_type='gaussian', intensity=intensity, seed=42)
        cv2.imshow("Gaussian Noise", gaussian_noise)
        cv2.imwrite("Gaussian_Noise_04_04.png", gaussian_noise)

        salt_pepper_noise = add_noise(image, noise_type='salt_pepper', intensity=intensity, seed=42)
        cv2.imshow("Salt & Pepper Noise", salt_pepper_noise)
        cv2.imwrite("Salt_Pepper_Noise.png", salt_pepper_noise)

        speckle_noise = add_noise(image, noise_type='speckle', intensity=intensity, seed=42)
        cv2.imshow("Speckle Noise", speckle_noise)
        cv2.imwrite("Speckle_Noise_04_04.png", speckle_noise)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Wystąpił błąd: {e}")
