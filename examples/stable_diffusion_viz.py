from pathlib import Path
from shutil import rmtree

from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import pyplot as plt

dir = Path("result")

bits = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
stds = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

images = list()
for y, bit in enumerate(reversed(bits)):
    for x, std in enumerate(stds):
        dest = dir / f"{bit}bit-{std}std.png"

        if not dest.exists():
            print("Not enough image!")
            exit(1)

        images.append(Image.open(dest))

fig = plt.figure(figsize=(8, 8))
grid = ImageGrid(fig, 111, nrows_ncols=(len(bits), len(stds)), axes_pad=0.1)
for ax, img in zip(grid, images):
    ax.imshow(img)
    ax.set_axis_off()
plt.savefig(dir / "grid-images.png", bbox_inches="tight", pad_inches=0.1)
