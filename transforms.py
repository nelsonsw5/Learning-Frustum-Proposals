import torch
import numpy as np
import random


class MeanShift(object):
    def __call__(self, pc, *args, **kwargs):
        mean = torch.mean(pc, dim=0)
        output = pc - mean
        return output


class NormalizeShift(object):
    def __init__(self):
        self.means = None
        self.max_dist = None

    def fit_transform(self, pc):
        self.means = torch.mean(pc, dim=0)
        translated = pc - self.means

        origin = torch.zeros(3)
        dist = torch.norm(translated - origin, p=2, dim=1)

        self.max_dist = torch.max(dist)
        output = translated / self.max_dist
        return output

    def __call__(self, pc):
        translated = pc - self.means
        output = translated / self.max_dist
        return output

    def scale(self, pc):
        output = pc / self.max_dist
        return output


class YForward2NegZForward(object):
    """ Transform from (Y forward, Z up) to (Z forward, -Y up)
        See visualization in docs:
            https://delicious-ai.atlassian.net/wiki/spaces/ML/pages/1241055233/Coordinate+systems
    """

    def __call__(self, verts, invert=True, *args, **kwargs):
        """

        @param verts: (torch.Tensor) point cloud (n x 3)
        @param invert: (bool) invert last dimensions (multiply by negative one)
        @param args:
        @param kwargs:
        @return:
        """
        x, y, z = verts.unbind(-1)
        if invert:
            swapped = [x.unsqueeze(-1), z.unsqueeze(-1), -y.unsqueeze(-1)]
        else:
            swapped = [x.unsqueeze(-1), z.unsqueeze(-1), y.unsqueeze(-1)]
        return torch.cat(swapped, -1)


class NegZForward2YForward(object):
    """Inverse of YForward2NegZForward"""

    def __call__(self, verts, invert=True, *args, **kwargs):
        x, y, z = verts.unbind(-1)
        if invert:
            swapped = [x.unsqueeze(-1), -z.unsqueeze(-1), y.unsqueeze(-1)]
        else:
            swapped = [x.unsqueeze(-1), z.unsqueeze(-1), y.unsqueeze(-1)]

        return torch.cat(swapped, -1)

class MaxNormalizer(object):
    def __init__(self, max, scale=True):

        if scale:
            assert max is not None, "max value must be provided"

        self.max = max
        self.scale = scale

    def __call__(self, x, *args, **kwargs):
        if self.scale:
            return x / self.max
        else:
            return x

class MaxDeNormalizer(object):
    def __init__(self, max):
        self.max = max
    def __call__(self, x, *args, **kwargs):
        return x * self.max


# class RandomRotationZ(object):
#     """ Rotate points around the Z axis """
#
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
#         self.angle = None
#
#
#     def fit_transform(self, pc):
#         self.angle = float(torch.normal(self.mean, self.std, (1, )))
#         rot = RotateAxisAngle(angle=self.angle,
#                               axis="Y",  # Swap axes operator swaps Z & Y axes in simulated data and
#                                          # for real data Z & Y axes are already swapped
#                               degrees=True)
#         return rot.transform_points(pc)
#
#
#     def __call__(self, pc):
#         rot = RotateAxisAngle(angle=self.angle,
#                               axis="Y",  # Swap axes operator swaps Z & Y axes in simulated data and
#                                          # for real data Z & Y axes are already swapped
#                               degrees=True)
#         return rot.transform_points(pc)


class AddNoise(object):
    """ Adds Noise to the point cloud """

    def __init__(self, mean, std, noise_percentage):
        self.mean = mean
        self.std = std
        self.noise_percentage = noise_percentage


    def __call__(self, pc):
        noise = torch.normal(self.mean, self.std, size=pc.shape)
        mask = torch.FloatTensor(pc.shape).uniform_() < self.noise_percentage
        return pc + (noise*mask)


class MirrorX(object):
    """ Mirrors pointclouds about the X axis """

    def __init__(self, probability):
        self.mirror = None
        self.probability = probability

    def fit_transform(self, pc):
        self.mirror = int(float(torch.rand(1)) < self.probability)
        if self.mirror:
            pc[:, 0] = -pc[:, 0]
        return pc


    def __call__(self, pc, *args, **kwargs):
        if self.mirror:
            pc[:, 0] = -pc[:, 0]
        return pc


class ShufflePoints(object):
    """
        randomly shuffle order of first dimension of tensor
            - fit_transform() should be called when the user wants
            to cache the random ordering
            _ __call__() should be used when the cached ordering exists

        For example, when we want shuffle both a point cloud and the semantic segmentation labels:
            shuffler = ShufflePoints()
            pc = shuffler.fit_transform(pc)
            sem_seg = shuffler(sem_seg)

    """

    def __init__(self):
        self.idx = None

    def fit_transform(self, pc):
        self.idx = np.arange(pc.shape[0])
        np.random.shuffle(self.idx)
        return pc[self.idx]

    def __call__(self, pc, *args, **kwargs):
        return pc[self.idx]


class GaussianResize(object):
    """
    Resize point cloud using Gaussian noise
    """

    def __init__(self, mu: float = 0.0, sig: float = 0.1, samples: float = 0.25, broadcast: bool = False):
        self.mu = mu + 1.0 # perturbation relative to existing points
        self.sig = sig
        self.samples = samples
        self.broadcast = broadcast

    def __call__(self, pc: np.ndarray, *args, **kwargs):
        """

        @param pc: input point cloud dims (batch_size, num_points, num_channels)
        @return:
        """
        alpha = random.random()

        if alpha >= self.samples:
            # don't perturb point cloud
            return pc

        if self.broadcast:

            bs, n, c = pc.shape
            ndim = pc.ndim
            shp = [-1] + [1]*(ndim-1)

            mu = pc.mean(axis=1).reshape(bs, 1, c)
            scalar = np.random.normal(self.mu, self.sig, size=bs)
            dist = pc-mu
            scalar = scalar.reshape(shp)
            output = scalar*dist + pc - dist
        else:
            scalar = np.random.normal(self.mu, self.sig)
            output = pc * scalar

        return output