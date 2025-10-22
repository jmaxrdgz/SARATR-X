import os
import argparse

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


def chip_sar_images(input_dir: str, chip_size: int = 512):
    # Create output
    output_dir = os.path.join(input_dir, f"chips_{chip_size}")
    os.makedirs(output_dir, exist_ok=True)

    # List tiff images
    tiff_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".tiff", ".tif"))]

    for fname in tqdm(tiff_files, desc="Images", unit="img"):
        fpath = os.path.join(input_dir, fname)
        with rasterio.open(fpath) as src:
            width, height = src.width, src.height

            base_name = os.path.splitext(fname)[0]

            coords = [
                (x, y)
                for y in range(0, height - chip_size + 1, chip_size)
                for x in range(0, width - chip_size + 1, chip_size)
            ]

            for x, y in tqdm(coords, desc=f"  Chips {fname}", leave=False, unit="chip"):
                window = Window(x, y, chip_size, chip_size)
                chip = src.read(1, window=window)

                # Capella normalization & float16 conversion
                chip = np.abs(chip)
                chip = np.log(np.maximum(chip, 1e-6))/16 # Capella log norm 
                chip = chip.astype(np.float16)

                # Chip name
                chip_name = f"{base_name}_{x}_{y}"
                chip_path = os.path.join(output_dir, chip_name)

                # Save
                np.save(chip_path, chip)

    print(f"Save chips at : {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chip SAR images from Capella")
    parser.add_argument("input_dir", type=str, help="SAR images folder (.tiff)")
    parser.add_argument("--chip_size", type=int, default=512, help="Chip size (default is 512)")
    args = parser.parse_args()

    chip_sar_images(args.input_dir, args.chip_size)