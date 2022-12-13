import pandas as pd
import torch
import numpy as np

from typing import Optional

def load_obj(fpath, get_normals=False, filter_high=False):
    rows = pd.read_csv(
        fpath,
        sep=" ",
        engine="c",
        header=None,
        comment='#'
    )
    verts = rows[rows[0] == "v"]

    if filter_high and verts.shape[1] == 5:
        verts = verts[verts[4] == 2.0]

    verts = torch.tensor(verts.iloc[:, 1:4].astype(np.float32).values)

    if get_normals:
        normals = rows[rows[0] == "vn"]
        normals = torch.tensor(normals.iloc[:, 1:].astype(np.float32).values)
    else:
        normals = None

    return verts, normals


def write_obj(
        verts: torch.Tensor,
        fpath: str,
        normals: Optional[torch.Tensor] = None,
):
    if isinstance(verts, torch.Tensor):
        verts = verts.data.numpy()

    vert_dict = {
        'header': ['v']*len(verts),
        'x': verts[:, 0],
        'y': verts[:, 1],
        'z': verts[:, 2]
    }

    output = pd.DataFrame.from_dict(
        vert_dict
    )

    if normals is not None:
        if isinstance(normals, torch.Tensor):
            normals = normals.data.numpy()

        normals_dict = {
            'header': ['vn'] * len(normals),
            'x': normals[:, 0],
            'y': normals[:, 1],
            'z': normals[:, 2]
        }

        normals = pd.DataFrame.from_dict(
            normals_dict
        )
        output = pd.concat([output, normals], axis=0)

    output.to_csv(
        fpath,
        index=False,
        header=False,
        sep=" ",
        float_format="%.5f"
    )