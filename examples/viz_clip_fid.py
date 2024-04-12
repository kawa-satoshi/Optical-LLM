import itertools

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.lines import Line2D


markers = itertools.cycle(Line2D.markers.keys())
linestyles = itertools.cycle(("solid", "dashed", "dashdot", "dotted"))

df = pd.read_csv("clip-score.csv", index_col=0)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

for index, row in df.iterrows():
    row.plot(ax=ax, label=index, linestyle=next(linestyles), marker=next(markers))

ax.grid()
ax.set_xlabel("Noise")
ax.set_ylabel("CLIP Score")
fig.legend(title="Bitwidth", loc="lower right")

fig.suptitle(
    "CLIP Score of CompVis/ldm-text2im-large-256 model with various bitwidth and noise\n(Original Score = 25.54, only applied to QKV layer in UNet)"
)
fig.savefig("clip-score.png")

# ===

df = pd.read_csv("fid-score.csv", index_col=0)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

for index, row in df.iterrows():
    row.plot(ax=ax, label=index, linestyle=next(linestyles), marker=next(markers))

ax.grid()
ax.set_xlabel("Noise")
ax.set_ylabel("FID Score")
fig.legend(title="Bitwidth", loc="lower right")

fig.suptitle(
    "FID Score of facebook/DiT-XL-2-256 model with various bitwidth and noise\n(Original Score = 163.80, only applied to QKV layer in transformer)"
)
fig.savefig("fid-score.png")
