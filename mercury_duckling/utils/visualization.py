import matplotlib.pyplot as plt
import numpy as np


def plot_image_and_mask(image, mask, ax, cmap="Greys", **mask_params):
    ax.imshow(image, cmap)
    show_mask(mask=mask, ax=ax, **mask_params)
    ax.axis("off")


def show_mask(
    mask,
    ax,
    color=np.array([30 / 255, 144 / 255, 255 / 255]),
    random_color=False,
    alpha=0.3,
):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.concatenate([color, np.array([alpha])], axis=0)
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="o",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
        alpha=0.7,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="X",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
        alpha=0.7,
    )


def show_box(x, y, h, w, ax):
    """Expects center point and height, width"""
    ax.add_patch(
        plt.Rectangle((x, y), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
