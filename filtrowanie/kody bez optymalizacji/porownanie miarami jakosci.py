import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def calculate_ssim(img1, img2):
    ssim_value, _ = ssim(img1, img2, full=True, channel_axis=-1, win_size=3)
    return ssim_value


def calculate_ncd(img1, img2):
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    L1, A1, B1 = cv2.split(img1_lab)
    L2, A2, B2 = cv2.split(img2_lab)
    delta_L = L1.astype(np.float32) - L2.astype(np.float32)
    delta_A = A1.astype(np.float32) - A2.astype(np.float32)
    delta_B = B1.astype(np.float32) - B2.astype(np.float32)
    delta_E = np.sqrt(delta_L ** 2 + delta_A ** 2 + delta_B ** 2)
    max_delta_e = np.sqrt(100 ** 2 + 255 ** 2 + 255 ** 2)
    normalized_delta_e = delta_E / max_delta_e
    return np.mean(normalized_delta_e)



def main():
    img1 = cv2.imread('kodim04.png')
    img2 = cv2.imread('output_gauss9_5.jpg')

    if img1 is None or img2 is None:
        raise FileNotFoundError("Jeden z obrazów nie został znaleziony.")

    if img1.shape != img2.shape:
        raise ValueError("Obrazy muszą mieć ten sam rozmiar!")

    psnr_value = calculate_psnr(img1, img2)
    ssim_value = calculate_ssim(img1, img2)
    ncd_value = calculate_ncd(img1, img2)

    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"NCD: {ncd_value:.4f}")


if __name__ == "__main__":
    main()
