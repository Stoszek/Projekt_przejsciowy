import math
from PIL import Image


def generate_gaussian_kernel(size, sigma):
    if size % 2 == 0:
        raise ValueError("Rozmiar jądra musi być nieparzysty.")
    kernel = [[0 for _ in range(size)] for _ in range(size)]
    center = size // 2
    sum_val = 0

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i][j] = (1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            sum_val += kernel[i][j]

    for i in range(size):
        for j in range(size):
            kernel[i][j] /= sum_val

    return kernel


def print_kernel(kernel):
    print("Jądro Gaussa:")
    for row in kernel:
        print(" ".join(f"{val:.5f}" for val in row))


def apply_gaussian_filter(image, kernel):
    width, height = image.size
    pixels = image.load()
    size = len(kernel)
    offset = size // 2

    output_image = Image.new("RGB", (width, height))
    output_pixels = output_image.load()

    for x in range(width):
        for y in range(height):
            acc_r, acc_g, acc_b = 0, 0, 0
            for i in range(size):
                for j in range(size):
                    xi = x + i - offset
                    yj = y + j - offset
                    if 0 <= xi < width and 0 <= yj < height:
                        r, g, b = pixels[xi, yj]
                        acc_r += r * kernel[i][j]
                        acc_g += g * kernel[i][j]
                        acc_b += b * kernel[i][j]
            output_pixels[x, y] = (
                int(min(max(acc_r, 0), 255)),
                int(min(max(acc_g, 0), 255)),
                int(min(max(acc_b, 0), 255))
            )
    return output_image


def main():
    kernel_size = 7
    sigma = 10.0
    input_image = Image.open("Salt_Pepper_Noise.png")
    gaussian_kernel = generate_gaussian_kernel(kernel_size, sigma)
    print_kernel(gaussian_kernel)
    blurred_image = apply_gaussian_filter(input_image, gaussian_kernel)
    blurred_image.save("output_gauss9_5.jpg")
    print("Obraz został przetworzony i zapisany jako 'output3.jpg'.")


if __name__ == "__main__":
    main()
