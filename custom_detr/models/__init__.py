from .detroap import build_DETR


def build_model(args):
    return build_DETR(args)