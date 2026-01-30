import numpy as np
from PIL import Image


def gaussian(x, sigma):
    return np.exp(-(x ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(image, spatial_sigma, intensity_sigma):
    image_array = np.array(image, dtype=np.float32)
    height, width, channels = image_array.shape
    kernel_size = int(6 * spatial_sigma) | 1
    offset = kernel_size // 2

    spatial_kernel = np.fromfunction(
        lambda x, y: gaussian(np.sqrt((x - offset) ** 2 + (y - offset) ** 2), spatial_sigma),
        (kernel_size, kernel_size),
        dtype=np.float32,
    )

    output_image = np.zeros_like(image_array, dtype=np.float32)
    padded_image = np.pad(image_array, ((offset, offset), (offset, offset), (0, 0)), mode="reflect")

    for y in range(height):
        for x in range(width):
            region = padded_image[y:y + kernel_size, x:x + kernel_size, :]
            center_intensity = padded_image[y + offset, x + offset, :]
            intensity_diff = region - center_intensity
            intensity_kernel = np.exp(-(intensity_diff ** 2) / (2 * intensity_sigma ** 2))
            combined_kernel = spatial_kernel[:, :, None] * intensity_kernel
            combined_kernel /= np.sum(combined_kernel, axis=(0, 1), keepdims=True)
            output_image[y, x, :] = np.sum(region * combined_kernel, axis=(0, 1))

    return Image.fromarray(np.clip(output_image, 0, 255).astype(np.uint8))


def main():
    spatial_sigma = 5.0
    intensity_sigma = 50.0
    input_image = Image.open("Salt_Pepper_Noise.png").convert("RGB")
    filtered_image = apply_bilateral_filter(input_image, spatial_sigma, intensity_sigma)
    filtered_image.save("bilateral_filtered_5_50.jpg")
    print("Obraz został przetworzony i zapisany.")


if __name__ == "__main__":
    main()
