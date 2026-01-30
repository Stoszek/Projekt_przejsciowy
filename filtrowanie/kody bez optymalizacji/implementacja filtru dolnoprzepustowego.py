from PIL import Image


def generate_mean_kernel(size):
    if size % 2 == 0:
        raise ValueError("Rozmiar jądra musi być nieparzysty.")
    return [[1 / (size * size) for _ in range(size)] for _ in range(size)]


def print_kernel(kernel):
    print("Jądro filtru uśredniającego:")
    for row in kernel:
        print(" ".join(f"{val:.5f}" for val in row))


def apply_filter(image, kernel):
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
    try:
        kernel_size = 9
        input_image = Image.open("kodim04.png")
        mean_kernel = generate_mean_kernel(kernel_size)
        print_kernel(mean_kernel)
        smoothed_image = apply_filter(input_image, mean_kernel)
        smoothed_image.save("output_mean9.jpg")
        print("Obraz został przetworzony i zapisany jako 'output_mean.jpg'.")
    except FileNotFoundError:
        print("Nie znaleziono pliku obrazu. Upewnij się, że plik istnieje w folderze.")
    except ValueError as e:
        print(f"Błąd: {e}")
    except Exception as e:
        print(f"Nieoczekiwany błąd: {e}")


if __name__ == "__main__":
    main()
