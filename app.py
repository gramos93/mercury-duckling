import matplotlib
import numpy as np

matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
matplotlib.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
matplotlib.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"

# from mercury_duckling.configs import SAM_THERMAL
# from mercury_duckling.pipelines.sam_interactive import SamInteractiveTest
import sys

import matplotlib.pyplot as plt
from rich.progress import track

sys.path.append("./mercury_duckling/models/")

from mercury_duckling.configs import RITM_THERMAL
from mercury_duckling.pipelines.ritm_interactive import RitmInteractiveTest

# sys.path.append("./mercury_duckling/models/inter_unet")

# from mercury_duckling.configs import UNET_THERMAL
# from mercury_duckling.pipelines.unet_interactive import UnetInteractiveTest

RITM_THERMAL["dataset"]["root"] = "../../data/_500_per_folder/merged_dataset_delam/"


def show_mask(
    mask, ax, color=np.array([30 / 255, 144 / 255, 255 / 255, 0.6]), random_color=False
):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=150):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="o",
        s=marker_size,
        edgecolor="black",
        linewidth=1.25,
        alpha=1,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="X",
        s=marker_size,
        edgecolor="black",
        linewidth=1.25,
        alpha=1,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def NOCS(ious, thresh, max_clicks=20):
    """Number of clicks to reach threshold"""
    nocs = []
    for i in range(ious.shape[0]):
        for j in range(max_clicks):
            if ious[i, j] >= thresh:
                nocs.append(j + 1)
                break
        if len(nocs) == i:
            nocs.append(max_clicks)
    return nocs


def iou_score(y_hat, y):
    return np.logical_and(y_hat, y).sum() / np.logical_or(y_hat, y).sum()


def test_pipeline(pipeline):
    ious = []
    for i in track(range(2), description="Testing"):
        image, target, id = pipeline._dataset[i]
        image = (image * 255).astype(np.uint8)
        target = target.transpose(2, 0, 1)
        for i, mask in enumerate(np.array(target)):
            mask_ious = [0.0]
            aux = None
            masks = None
            for prompts in pipeline.sampler.interact(mask):
                masks, aux = pipeline.predict(
                    image, prompts, aux if aux is not None else masks, id
                )
                pipeline.sampler.set_outputs(masks)
                mask_ious.append(iou_score(masks, mask))
            ious.append(mask_ious)
    return ious


masks_color = np.array([30 / 225, 225 / 225, 100 / 225, 0.6])

# sam_pipeline = SamInteractiveTest(SAM_THERMAL)
# sam_pipeline.on_experiment_start(sam_pipeline)

ritm_pipeline = RitmInteractiveTest(RITM_THERMAL)
ritm_pipeline.on_experiment_start(ritm_pipeline)

# unet_pipeline = UnetInteractiveTest(UNET_THERMAL)
# unet_pipeline.on_experiment_start(unet_pipeline)


if __name__ == "__main__":
    ious = test_pipeline(ritm_pipeline)  # ritm_pipeline, unet_pipeline))
    means = np.array(ious).mean(axis=0)
    np.savetxt("ritm_means.csv", means, delimiter=",")

    # fig, ax = plt.subplots(figsize=(10, 5))
    # names =["SAM", "RITM", "UNet"]
    # ax.plot(means)
    # ax.set_xlabel("Number of clicks")
    # ax.set_ylabel("Intersection Over Union")
    # ax.set_xticks(range(21))
    # ax.set_yticks(np.arange(0, 1.1, 0.1))
    # ax.set_ylim([0, 1])
    # ax.set_xlim([0, 20.7])
    # ax.spines[['right', 'top']].set_visible(False)
    # plt.legend(names)
    # plt.grid(axis='y', linestyle='--')
    # plt.savefig("NoC.jpg")
    # plt.show()
