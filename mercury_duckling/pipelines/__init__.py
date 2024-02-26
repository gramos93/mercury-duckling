from .core import InteractiveTest, SegmentationExp


def build_pipeline(cfg, model, dataset):
    if cfg.model.type == "interactive":
        return InteractiveTest(predictor=model, dataset=dataset, config=cfg)
    else:
        return SegmentationExp(segmentor=model, dataset=dataset, config=cfg)
