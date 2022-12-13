from typing import List, Dict

import torch
from datasets.dataset_utils import get_zero_padded_batch



def collate_density_reg(batch: List[Dict]):
    _keys_to_stack = set(["num_targets", "points"])
    _list_keys_flatten = set(["names", "upcs"])
    _list_keys = set(["img"])

    if batch is None or len(batch) == 0:
        return None
    x = {}

    collated_dict = {
        "targets": []
    }
    for b in batch:
        inputs, target = b
        collated_dict["targets"].append(target)
        for k, v in inputs.items():
            if k not in collated_dict:
                collated_dict[k] = []
            collated_dict[k].append(v)

    points = []
    for p in collated_dict["points"]:
        for p_i in p:
            points.append(p_i)
    collated_dict["points"] = points

    for k, v in collated_dict.items():
        if k in _keys_to_stack:
            x[k] = torch.stack(v, dim=0)
        elif k in _list_keys_flatten:
            x[k] = []
            for l in v:
                x[k] += l
        elif k in _list_keys:
            x[k] = []
            for l in v:
                x[k].append(l)
        else:
            x[k] = torch.cat(v)

    return x, x['targets']

def collate_density_padded(batch: List[Dict]):
    _keys_to_stack = set(["num_targets"])
    _list_keys_flatten = set(["names", "upcs"])
    _list_keys = set(["img"])
    _zero_pad_keys = set(["targets", "points", "object_geo_type"])

    if batch is None or len(batch) == 0:
        return None
    x = {}

    collated_dict = {
        "targets": []
    }
    for b in batch:
        inputs, target = b
        collated_dict["targets"].append(target)
        for k, v in inputs.items():
            if k not in collated_dict:
                collated_dict[k] = []
            collated_dict[k].append(v)

    for k in _zero_pad_keys:
        if k == "targets":
            x[k] = get_zero_padded_batch(collated_dict[k], flatten_last=True)
        else:
            x[k] = get_zero_padded_batch(collated_dict[k])

    for k, v in collated_dict.items():
        if k in _zero_pad_keys:
            continue
        if k in _keys_to_stack:
            x[k] = torch.stack(v, dim=0)
        elif k in _list_keys_flatten:
            x[k] = []
            for l in v:
                x[k] += l
        elif k in _list_keys:
            x[k] = []
            for l in v:
                x[k].append(l)
        else:
            x[k] = torch.cat(v)
    return x, x['targets']


def get_collate_fn(model_type: str):
    if model_type == "pointnet_transformer" or model_type == "pointnet2_transformer":
        return collate_density_padded
    else:
        return collate_density_reg