import os
import click
from omegaconf import OmegaConf
from mercury_duckling.datasets import build_segmentation, build_thermal

from mercury_duckling.utils import console
from mercury_duckling.models import build_predictor, build_segmentor


@click.command()
@click.option("--device", default=None, help="Specify the device to use.")
@click.option(
    "--model",
    default=None,
    help="Specify the model to use. One of [`sam`, `ritm`, `iunet`].",
)
@click.option(
    "--dataset",
    default=None,
    help="Specify the dataset to use. One of [`thermal`, `ape`].",
)
@click.option(
    "--mode",
    default="train",
    help="Mode of operation. One of [`train`, `test`, `infer`].",
)
def main(device, model, dataset, mode):
    console.rule("[bold]Mercury Duckling Pipeline.")
    cfg_base = OmegaConf.load("configs/base.yaml")
    cfg_data = OmegaConf.load("configs/dataset.yaml")
    cfg_model = OmegaConf.load("configs/model.yaml")

    cfg_base.device = device if device is not None else cfg_base.device

    if dataset is not None and (data := cfg_data.dataset.get(dataset, False)):
        cfg_data.default = data
    elif not data:
        console.log("[bold red]Invalid dataset. Exiting.")
        return 1

    if model is not None and (model := cfg_model.models.get(model, False)):
        cfg_model.selected_model = model
    elif not model:
        console.log("[bold red]Invalid model. Exiting.")
        return 2

    if mode in ["train", "test", "infer"]:
        cfg_base.mode = mode
    else:
        console.log("[bold red]Invalid mode. Exiting.")
        return 3

    cfg = OmegaConf.merge(cfg_base, cfg_data, cfg_model)
    console.log("[bold green]Configuration generated. Building pipeline...")
    model = (
        build_predictor(cfg)
        if cfg.model.type == "interactive"
        else build_segmentor(cfg)
    )
    dataset = (
        build_thermal(cfg)
        if cfg.selected_data == "thermal"
        else build_segmentation(cfg)
    )
    # exp =
    return 0


if __name__ == "__main__":
    main()
