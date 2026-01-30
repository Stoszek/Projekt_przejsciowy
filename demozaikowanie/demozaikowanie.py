from pathlib import Path

import numpy as np
import statistics
from skimage import color, io
import cv2

class Demosaic:
  def __init__(self, original_image: np.ndarray):
    self.original_image = original_image
    h, w, _ = original_image.shape
    self.height = h
    self.width = w
    self.working_image = np.zeros((h, w, 3), dtype=np.float32)
    self.pattern = np.zeros((h, w), dtype=np.int16)

  def create_bayer_pattern(self, pattern_name="GBRG"):
    patterns = {
      "GBRG": [(1, 2), (0, 1)],
      "GRBG": [(1, 0), (2, 1)],
      "BGGR": [(2, 1), (1, 0)],
      "RGGB": [(0, 1), (1, 2)],
    }

    pattern = patterns.get(pattern_name)

    self.working_image[::2, ::2, pattern[0][0]] = self.original_image[::2, ::2, pattern[0][0]]
    self.working_image[::2, 1::2, pattern[0][1]] = self.original_image[::2, 1::2, pattern[0][1]]
    self.working_image[1::2, ::2, pattern[1][0]] = self.original_image[1::2, ::2, pattern[1][0]]
    self.working_image[1::2, 1::2, pattern[1][1]] = self.original_image[1::2, 1::2, pattern[1][1]]

    self.pattern[::2, ::2] = pattern[0][0]
    self.pattern[::2, 1::2] = pattern[0][1]
    self.pattern[1::2, ::2] = pattern[1][0]
    self.pattern[1::2, 1::2] = pattern[1][1]

    return self.working_image, self.pattern

  def create_fuji_pattern(self):
    pattern_matrix = [
      [1, 0, 2, 1, 2, 0],  # G B R G R B
      [2, 1, 1, 0, 1, 1],  # R G G B G G
      [0, 1, 1, 2, 1, 1],  # B G G R G G
      [1, 2, 0, 1, 0, 2],  # G R B G B R
      [0, 1, 1, 2, 1, 1],  # B G G R G G
      [2, 1, 1, 0, 1, 1],  # R G G B G G
    ]

    for row in range(6):
      for col in range(6):
        channel = pattern_matrix[row][col]
        self.working_image[row::6, col::6, channel] = self.original_image[row::6, col::6, channel]
        self.pattern[row::6, col::6] = channel

    return self.working_image, self.pattern

  def nearest_neighbor(self):
    for x in range(1, self.height + 1):
      for y in range(1, self.width + 1):
        for ii in range(x - 1, x + 2):
          for jj in range(y - 1, y + 2):
            if self.pattern[x, y] == self.pattern[ii, jj]:
              continue
            match self.pattern[x, y]:
              case 0:
                if self.pattern[ii, jj] == 1:
                  self.working_image[x, y, 1] = self.working_image[ii, jj, 1]
                elif self.pattern[ii, jj] == 2:
                  self.working_image[x, y, 2] = self.working_image[ii, jj, 2]
              case 1:
                if self.pattern[ii, jj] == 0:
                  self.working_image[x, y, 0] = self.working_image[ii, jj, 0]
                elif self.pattern[ii, jj] == 2:
                  self.working_image[x, y, 2] = self.working_image[ii, jj, 2]
              case 2:
                if self.pattern[ii, jj] == 0:
                  self.working_image[x, y, 0] = self.working_image[ii, jj, 0]
                elif self.pattern[ii, jj] == 1:
                  self.working_image[x, y, 1] = self.working_image[ii, jj, 1]
    return self.working_image[1:-1, 1:-1]

  def bilinear(self):
    for x in range(1, self.height + 1):
      for y in range(1, self.width + 1):
        red, green, blue = [], [], []
        for ii in range(x - 1, x + 2):
          for jj in range(y - 1, y + 2):
            if self.pattern[x, y] == self.pattern[ii, jj]:
              continue
            else:
              match self.pattern[x][y]:
                case 0:
                  if self.pattern[ii, jj] == 1:
                    green.append(self.working_image[ii, jj, 1])
                  elif self.pattern[ii, jj] == 2:
                    blue.append(self.working_image[ii, jj, 2])
                case 1:
                  if self.pattern[ii, jj] == 0:
                    red.append(self.working_image[ii, jj, 0])
                  elif self.pattern[ii, jj] == 2:
                    blue.append(self.working_image[ii, jj, 2])
                case 2:
                  if self.pattern[ii, jj] == 0:
                    red.append(self.working_image[ii, jj, 0])
                  elif self.pattern[ii, jj] == 1:
                    green.append(self.working_image[ii, jj, 1])
        if red: self.working_image[x, y, 0] = statistics.fmean(red)
        if green: self.working_image[x, y, 1] = statistics.fmean(green)
        if blue: self.working_image[x, y, 2] = statistics.fmean(blue)
    return self.working_image[1:-1, 1:-1]

  def calculate_psnr(self, original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
      return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

  def calculate_ncd(self, original, processed):
    original_lab = color.rgb2lab(original / 255.0)
    processed_lab = color.rgb2lab(processed / 255.0)
    delta_E = np.sqrt(np.sum((original_lab - processed_lab) ** 2, axis=2))
    E_star_lab = np.sqrt(np.sum(original_lab ** 2, axis=2))
    ncd = np.sum(delta_E) / np.sum(E_star_lab)
    return ncd

def main():
  input_dir = Path("kodak_dataset")
  output_dir = Path("wyjscie")
  output_dir.mkdir(parents=True, exist_ok=True)

  pat = "GBRG"
  method = "bilinear"

  files = sorted(input_dir.glob("*.png"))

  psnrs = []
  ncds = []

  for in_path in files:
    img = io.imread(str(in_path)).astype(np.float32)

    ctx = Demosaic(img)

    if pat == "FUJI":
      ctx.create_fuji_pattern()
    else:
      ctx.create_bayer_pattern(pattern_name=pat)

    ctx.working_image = cv2.copyMakeBorder(ctx.working_image, 1, 1, 1, 1, borderType=cv2.BORDER_REPLICATE)
    ctx.pattern = cv2.copyMakeBorder(ctx.pattern, 1, 1, 1, 1, borderType=cv2.BORDER_REPLICATE)

    if method == "nearest":
      out = ctx.nearest_neighbor()
    else:
      out = ctx.bilinear()

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    out_path = output_dir / f"{in_path.stem}_demosaic_{pat}_{method}.png"
    io.imsave(str(out_path), out_u8)


    psnr = ctx.calculate_psnr(ctx.original_image, out.astype(np.float32))
    ncd = ctx.calculate_ncd(ctx.original_image, out.astype(np.float32))
    psnrs.append(psnr)
    ncds.append(ncd)
    print(f"{in_path.name}: PSNR={psnr:.4f} dB, NCD={ncd:.6f} -> {out_path.name}")

if __name__ == "__main__":
  main()
