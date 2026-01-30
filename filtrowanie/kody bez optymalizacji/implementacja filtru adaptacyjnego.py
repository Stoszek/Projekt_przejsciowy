import numpy as np
from PIL import Image


def median_filter_custom(image, size):
    pad_size = size // 2
    padded_image = np.pad(image, pad_size, mode='symmetric')
    filtered_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + size, j:j + size]
            filtered_image[i, j] = np.median(region)

    return filtered_image


def convolve(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='symmetric')
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            result[i, j] = np.sum(region * kernel)

    return result


def compute_local_stats(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    local_mean = convolve(image, kernel)
    squared_diff = (image - local_mean) ** 2
    local_variance = convolve(squared_diff, kernel)
    return local_mean, local_variance


def estimate_noise_variance(image, kernel_size):
    smoothed_image = median_filter_custom(image, size=kernel_size)
    noise = image - smoothed_image
    return np.var(noise)


def apply_adaptive_filter(image, noise_variance, kernel_size):
    local_mean, local_variance = compute_local_stats(image, kernel_size)
    filtered_image = image - (noise_variance / np.maximum(local_variance, noise_variance)) * (image - local_mean)
    return np.clip(filtered_image, 0, 255).astype(np.uint8)


def main():
    kernel_size = 3
    median_kernel_size = 9
    input_image = Image.open("kodim04.png").convert("RGB")
    image_array = np.array(input_image, dtype=np.float32)

    filtered_channels = []
    for channel in range(3):
        channel_array = image_array[:, :, channel]
        noise_variance = estimate_noise_variance(channel_array, kernel_size=median_kernel_size)
        print(f"Oszacowana wariancja szumu dla kanału {channel}: {noise_variance}")
        filtered_channel = apply_adaptive_filter(channel_array, noise_variance, kernel_size)
        filtered_channels.append(filtered_channel)

    filtered_image_array = np.stack(filtered_channels, axis=2)
    filtered_image = Image.fromarray(filtered_image_array.astype(np.uint8))
    filtered_image.save("adaptive_9_9x.jpg")
    print("Obraz został przetworzony i zapisany.")


if __name__ == "__main__":
    main()
