from enum import Enum

import torch
from utils import read_json


def prune_state_dict(state_dict):
    """
    Remove 'module.' from the names of all keys in state dict
    @param state_dict:
    @return:
    """
    pruned = {}

    for name, weights, in state_dict.items():

        if "module." in name:
            pruned_name = name.replace("module.", "")
        else:
            pruned_name = name

        pruned[pruned_name] = weights

    return pruned

def resolve_model_state(chkp: dict, model: dict):
    keep = {}
    drop = {}


    for name, param in chkp.items():
        # check if param exists
        if name in model:
            # check shape
            if param.shape == model[name].shape:
                keep[name] = param
            else:
                drop[name] = f"shape mismatch: {param.shape} != {model[name].shape}"
        else:
            drop[name] = f"parameter not defined: {name}"

    if drop:
        print("Removed layers:")
        for name, err in drop.items():
            print(f"\t-{name}: {err}")

    return keep


def load_model_chkp(model, chkp_path, use_cuda, strict=False):
    """

    @param model: torch model object
    @param chkp_path: (str) path to model checkpoint
    @param use_cuda: (str) pytorch device
    @param strict: (bool) strict enforcement of state dict matching to model
    @return:
    """

    """
    consider filtering to matching keys:
    ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                             if k in model.state_dict() and not any(x in k for x in exclude)
                             and model.state_dict()[k].shape == v.shape}
    
    """
    print(f"Loading weights from: {chkp_path}")
    device = torch.device('cuda' if use_cuda else 'cpu')
    chkp = torch.load(chkp_path, map_location=device)
    chkp_state = prune_state_dict(chkp["model_state_dict"])
    chkp_state = resolve_model_state(chkp_state, model.state_dict())
    model.load_state_dict(chkp_state, strict=strict)
    return model

def get_max_normalizer(fpath: str):
    normalizer = read_json(fpath)
    return normalizer["max"]

class NeonetTypes(Enum):
    REGRESSOR = "regressor"
    CLASSIFIER = "classifier"
    POISSON = 'poisson'


class EncoderTypes(Enum):
    POINTNET_TRANSFORMER = "pointnet_transformer"
    POINTNET2_TRANSFORMER = "pointnet2_transformer"
    POINTNET = "pointnet"
    POINTNET2 = "pointnet2"