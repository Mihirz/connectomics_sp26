"""Generate a stylized pixelated Morris Water Maze image for a conference poster.

Renders a 20x20 grid scene matching the environment's visual conventions
(blue circular pool, four colored landmark cues, hidden gold platform,
green agent), upscaled with nearest-neighbor for a crisp pixel-art look.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

GRID = 20
PIXEL_SCALE = 40

BG = np.array([0.06, 0.06, 0.10])
WATER = np.array([0.15, 0.25, 0.60])
WATER_HI = np.array([0.22, 0.36, 0.78])
WATER_LO = np.array([0.10, 0.18, 0.45])
PLATFORM = np.array([1.00, 0.85, 0.00])
PLATFORM_GLOW = np.array([1.00, 0.95, 0.45])
AGENT = np.array([0.10, 0.90, 0.30])
AGENT_OUT = np.array([0.00, 0.50, 0.10])
LANDMARKS = {
    "N": np.array([0.95, 0.25, 0.25]),
    "S": np.array([0.30, 0.65, 1.00]),
    "E": np.array([0.95, 0.55, 0.10]),
    "W": np.array([0.70, 0.35, 0.95]),
}


def in_pool(x, y, cx, cy, r):
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2


def render():
    img = np.tile(BG, (GRID, GRID, 1))
    cx, cy = (GRID - 1) / 2, (GRID - 1) / 2
    radius = 8.5

    rng = np.random.default_rng(7)
    for y in range(GRID):
        for x in range(GRID):
            if in_pool(x, y, cx, cy, radius):
                t = rng.random()
                shimmer = WATER + (WATER_HI - WATER) * (t ** 3) * 0.6
                ripple = 0.5 + 0.5 * np.sin((x + y) * 0.9 + y * 0.4)
                img[y, x] = WATER_LO + (shimmer - WATER_LO) * (0.55 + 0.45 * ripple)

    plat_x, plat_y = 13, 7
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            xx, yy = plat_x + dx, plat_y + dy
            if 0 <= xx < GRID and 0 <= yy < GRID and in_pool(xx, yy, cx, cy, radius):
                if dx == 0 and dy == 0:
                    img[yy, xx] = PLATFORM
                elif abs(dx) + abs(dy) == 1:
                    img[yy, xx] = PLATFORM * 0.75 + img[yy, xx] * 0.25
                else:
                    img[yy, xx] = PLATFORM_GLOW * 0.35 + img[yy, xx] * 0.65

    landmark_pos = {
        "N": (10, 1),
        "S": (10, 18),
        "E": (18, 10),
        "W": (1, 10),
    }
    for k, (lx, ly) in landmark_pos.items():
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                xx, yy = lx + dx, ly + dy
                if 0 <= xx < GRID and 0 <= yy < GRID:
                    if dx == 0 and dy == 0:
                        img[yy, xx] = LANDMARKS[k]
                    elif abs(dx) + abs(dy) == 1:
                        img[yy, xx] = LANDMARKS[k] * 0.6 + img[yy, xx] * 0.4

    agent_x, agent_y = 5, 14
    img[agent_y, agent_x] = AGENT
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        xx, yy = agent_x + dx, agent_y + dy
        if 0 <= xx < GRID and 0 <= yy < GRID:
            img[yy, xx] = AGENT_OUT * 0.55 + img[yy, xx] * 0.45

    trail = [(5, 14), (6, 13), (6, 12), (7, 12), (8, 11), (9, 11), (10, 10)]
    for i, (tx, ty) in enumerate(trail[:-1]):
        alpha = 0.18 + 0.05 * i
        img[ty, tx] = AGENT * alpha + img[ty, tx] * (1 - alpha)

    return img


def save_poster_image(out_path):
    img = render()
    upscaled = np.kron(img, np.ones((PIXEL_SCALE, PIXEL_SCALE, 1)))

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0a0a12")
    ax.imshow(upscaled, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#FFD700")
        spine.set_linewidth(3)

    ax.set_title("Morris Water Maze — 20x20 Grid-World Task",
                 color="#FFD700", fontsize=20, fontweight="bold", pad=18,
                 family="monospace")

    legend_items = [
        ("Agent", AGENT),
        ("Hidden Platform", PLATFORM),
        ("Pool (water)", WATER),
        ("Landmark Cues", np.array([0.85, 0.45, 0.55])),
    ]
    for i, (label, color) in enumerate(legend_items):
        ax.add_patch(Rectangle((20, 20 + i * 60), 40, 40,
                               facecolor=color, edgecolor="#FFD700", linewidth=1.5))
        ax.text(75, 50 + i * 60, label, color="white", fontsize=14,
                family="monospace", va="center")

    ax.text(upscaled.shape[1] - 20, upscaled.shape[0] - 20,
            "Augmented (PFC)  vs  Baseline\n+45.7% success",
            color="#FFD700", fontsize=14, fontweight="bold",
            family="monospace", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#0a0a12",
                      edgecolor="#FFD700", linewidth=1.5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#0a0a12")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    save_poster_image("results/poster_morris_water_maze.png")
