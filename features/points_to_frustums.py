"""
soure code: https://github.com/poodarchu/Det3D/blob/master/det3d/ops/point_cloud/point_cloud_ops.py
"""

import numpy as np
import torch
import numba
import math
from typing import Optional

from transforms import GaussianResize


class FrustumFeatures(object):

    def __init__(
            self,
            beam_dim,
            max_points=35,
            pc_range=(-1, -1, -1, 1, 1, 1),
            max_voxels=20000,
            epsilon=0.1,
            with_distance=False,
            zero_pad=True,
            augmented=False,
            local=False,
            resize: Optional[GaussianResize] = None
    ):
        """

        @param beam_dim: beam index over which to compute beams {0, 1, 2}
        @param max_points: max points in each voxel
        @param pc_range: point cloud range
        @param max_voxels: maximum number of voxels
        @param epsilon: (float), pct perturbation of bounding boxes
        @param with_distance: use distance feature (9-dims)
        @param zero_pad: zero_pad tensor
        @param augmented: augented features (distance to center, pillar offset, distance)
        """
        assert beam_dim in [0, 1, 2], "beam_dim must be 0, 1, 2"
        self.beam_dim = beam_dim
        self.pc_range = pc_range
        self.max_voxels = max_voxels
        self.max_points = max_points
        self.epsilon = epsilon
        self.augmented = augmented

        self.with_distance = with_distance
        self.zero_pad = zero_pad
        self.local = local
        self.resize = resize
