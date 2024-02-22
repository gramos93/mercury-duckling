import os
import click
from omegaconf import OmegaConf

from .pipelines import InteractiveTest
from .datasets import build_segmentation, build_thermal
from .utils import console
from .models import build_predictor, build_segmentor


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def add_dir(x):
    return os.path.join(THIS_DIR, x)


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
    cfg_base = OmegaConf.load(add_dir("configs/base.yaml"))
    cfg_data = OmegaConf.load(add_dir("configs/dataset.yaml"))
    cfg_model = OmegaConf.load(add_dir("configs/model.yaml"))

    cfg_base.device = device if device is not None else cfg_base.device

    if dataset is not None:
        data = cfg_data.datasets.get(dataset, False)
        if not data:
            console.log("[bold red]Invalid dataset. Exiting.")
            return 1
        else:
            cfg_data.selected_data = data

    if model is not None:
        model = cfg_model.models.get(model, False)
        if not model:
            console.log("[bold red]Invalid model. Exiting.")
            return 1
        else:
            cfg_model.selected_model= model

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
    console.log("[bold green]Pipeline built. Running experiment...")
    exp = InteractiveTest(
        predictor=model,
        dataset=dataset,
        config=cfg,
    )
    exp.run(exp)
    return 0


if __name__ == "__main__":
    main()
