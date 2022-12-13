"""
soure code: https://github.com/poodarchu/Det3D/blob/master/det3d/ops/point_cloud/point_cloud_ops.py
"""

import numpy as np
import torch
import numba
import math
import pdb


@numba.jit(nopython=True)
def _points_to_voxel_kernel(
        points,
        voxel_map,
        centroids,
        coors_range,
        num_points_per_voxel,
        voxels,
        voxel_size,
        grid_size,
        max_points=35
):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]

    for i in range(N):
        failed = False
        voxel_matches = np.argwhere((voxel_map[:, 0] <= points[i, 0]) & (voxel_map[:, 3] > points[i, 0]) &
                                    (voxel_map[:, 1] <= points[i, 1]) & (voxel_map[:, 4] > points[i, 1]) &
                                    (voxel_map[:, 2] <= points[i, 2]) & (voxel_map[:, 5] > points[i, 2]))
        if not voxel_matches.shape[0]:
            continue

        for idx, voxel_match in enumerate(voxel_matches):
            coor = np.floor((centroids[voxel_match,][0] - coors_range[:3]) / voxel_size).astype(np.int32)
            if np.any((coor<0) | (coor>=grid_size)) | (not voxel_match.shape[0]):
                failed = True
            if failed:
                continue

            voxelidx = int(voxel_match[0])
            num = num_points_per_voxel[voxelidx]
            if num < max_points:
                voxels[voxelidx, num] = points[i]
                num_points_per_voxel[voxelidx] += 1


class BeamFeatures(object):

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
            local=False
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

    @staticmethod
    def points_to_voxel(points,
                        beam_dim,
                        voxel_map,
                        centroids,
                        min_grid_size,
                        coors_range,
                        max_points=35,
                        max_voxels=20000):
        """convert kitti points(N, >=3) to voxels. This version calculate
        everything in one loop. now it takes only 4.2ms(complete point cloud)
        with jit and 3.2ghz cpu.(don't calculate other features)
        Note: this function in ubuntu seems faster than windows 10.

        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points and
                points[:, 3:] contain other information such as reflectivity.
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
            coors_range: [6] list/tuple or array, float. indicate voxel range.
                format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel.
            reverse_index: boolean. indicate whether return reversed coordinates.
                if points has xyz format and reverse_index is True, output
                coordinates will be zyx format, but points in features always
                xyz format.
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. you should shuffle points

        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points.
            coordinates: [M, 3] int32 tensor.
            num_points_per_voxel: [M] int32 tensor.
        """
        if not isinstance(voxel_map, np.ndarray):
            voxel_map = np.array(voxel_map, dtype=points.dtype)
        if not isinstance(coors_range, np.ndarray):
            coors_range = np.array(coors_range, dtype=points.dtype)


        # don't create large array in jit(nopython=True) code.
        num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)

        voxels = np.zeros(
            shape=(max_voxels, max_points, points.shape[-1]), dtype=np.float32)
        grid_count = int(math.ceil((coors_range[3] - coors_range[0]) / min_grid_size))

        voxel_size = []
        grid_size = ()
        for idx in range(3):
            if idx == beam_dim:
                voxel_size.append((coors_range[idx+3] - coors_range[idx]))
                grid_size += (1, )
            else:
                voxel_size.append((coors_range[idx+3] - coors_range[idx]) / grid_count)
                grid_size += (grid_count, )

        voxel_size = np.array(voxel_size)
        grid_size = np.array(grid_size).astype(np.int32)

        _points_to_voxel_kernel(
            points=points,
            voxel_map=voxel_map,
            centroids=centroids,
            coors_range=coors_range,
            num_points_per_voxel=num_points_per_voxel,
            voxels=voxels,
            max_points=max_points,
            voxel_size=voxel_size,
            grid_size=grid_size
        )
        return voxels, num_points_per_voxel, voxel_size

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        """Create boolean mask by actually number of a padded tensor.
        Args:
            actual_num ([type]): [description]
            max_num ([type]): [description]
        Returns:
            [type]: [description]
        """

        actual_num = torch.unsqueeze(actual_num, axis + 1)
        # tiled_actual_num: [N, M, 1]
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(
            max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]
        return paddings_indicator

    @staticmethod
    def xyzwlh2xyzxyz(xyzwlh, pc_range, epsilon, beam_dim):
        """

        @param xyzwlh:
        @param pc_range:
        @param epsilon:
        @param beam_dim:
        @return:
        """
        # Convert nx6 boxes from [x, y, z, w, l, h] to [x1, y1, z1, x2, y2, z2]
        # where xyz1=bottom-front-left, xyz2=top-back-right
        if beam_dim == 0:
            xyzxyz = np.zeros(xyzwlh.shape, dtype=xyzwlh.dtype)
            xyzxyz[:, 0] = pc_range[0]
            xyzxyz[:, 1] = xyzwlh[:, 1] - (xyzwlh[:, 4] * (1 + epsilon)) / 2
            xyzxyz[:, 2] = xyzwlh[:, 2] - (xyzwlh[:, 5] * (1 + epsilon)) / 2  # bottom front left z
            xyzxyz[:, 3] = pc_range[3]
            xyzxyz[:, 4] = xyzwlh[:, 1] + (xyzwlh[:, 4] * (1 + epsilon)) / 2  # top back right z
            xyzxyz[:, 5] = xyzwlh[:, 2] + (xyzwlh[:, 5] * (1 + epsilon)) / 2  # top back right z

        elif beam_dim == 1:
            xyzxyz = np.zeros(xyzwlh.shape, dtype=xyzwlh.dtype)
            xyzxyz[:, 0] = xyzwlh[:, 0] - (xyzwlh[:, 3] * (1+epsilon)) / 2  # bottom front left x
            xyzxyz[:, 1] = pc_range[1]                                      # bottom front left y
            xyzxyz[:, 2] = xyzwlh[:, 2] - (xyzwlh[:, 5] * (1+epsilon)) / 2  # bottom front left z
            xyzxyz[:, 3] = xyzwlh[:, 0] + (xyzwlh[:, 3] * (1+epsilon)) / 2  # top back right x
            xyzxyz[:, 4] = pc_range[4]                                      # top back right y
            xyzxyz[:, 5] = xyzwlh[:, 2] + (xyzwlh[:, 5] * (1+epsilon)) / 2  # top back right z
        elif beam_dim == 2:
            xyzxyz = np.zeros(xyzwlh.shape, dtype=xyzwlh.dtype)
            xyzxyz[:, 0] = xyzwlh[:, 0] - (xyzwlh[:, 3] * (1 + epsilon)) / 2  # bottom front left x
            xyzxyz[:, 1] = xyzwlh[:, 1] - (xyzwlh[:, 4] * (1 + epsilon)) / 2
            xyzxyz[:, 2] = pc_range[2]
            xyzxyz[:, 3] = xyzwlh[:, 0] + (xyzwlh[:, 3] * (1 + epsilon)) / 2  # top back right x
            xyzxyz[:, 4] = xyzwlh[:, 1] + (xyzwlh[:, 4] * (1 + epsilon)) / 2  # top back right z
            xyzxyz[:, 5] = pc_range[5]
        else:
            raise NotImplementedError(f"idx = {beam_dim} not supported")

        return xyzxyz

    @staticmethod
    def _get_min_grid_size(dimensions, beam_dim):
        indices = list(range(3))
        _ = indices.pop(beam_dim)
        dims = []
        for idx in indices:
            dims.append(np.min(dimensions[:, idx]))
        return float(min(dims))

    @staticmethod
    def get_dist_feats(feats, beam_dim):
        """
        Get univariate distance to extreme values (min and max) within a beam. I
        Intuitively, this gives the model information about how far the each point is from the "front" and the "back"
        @param feats: (torch.Tensor) tensor with dims (n_beams, n_points, 3)
        @param beam_dim: (int) dimension ({0,1,2}) along which to compute distance
        @return: (torch.Tensor) resulting distance feature with dims (n_beams, n_points, 2)
        """
        max_points = feats[:, :, :3].max(dim=1, keepdim=True)[0]
        min_points = feats[:, :, :3].min(dim=1, keepdim=True)[0]

        dist_feats = torch.cat(
            [
                (max_points[:, :, beam_dim] - feats[:, :, beam_dim]).unsqueeze(-1),
                (min_points[:, :, beam_dim] - feats[:, :, beam_dim]).unsqueeze(-1)
            ],
            dim=-1
        )

        return dist_feats

    def __call__(self, points, centroids, dimensions, *args, **kwargs):
        pdb.set_trace()

        if isinstance(points, torch.Tensor):
            points = points.data.numpy()

        if isinstance(centroids, torch.Tensor):
            centroids = centroids.data.numpy()

        if isinstance(dimensions, torch.Tensor):
            dimensions = dimensions.data.numpy()

        voxel_map = self.xyzwlh2xyzxyz(np.hstack([centroids, dimensions]), self.pc_range, self.epsilon, self.beam_dim)
        min_grid_size = self._get_min_grid_size(dimensions, self.beam_dim)

        features, num_voxels, voxel_size = self.points_to_voxel(
            points=points,
            beam_dim=self.beam_dim,
            voxel_map=voxel_map,
            centroids=centroids,
            min_grid_size=min_grid_size,
            coors_range=np.array(list(self.pc_range)),
            max_points=self.max_points,
            max_voxels=int(self.max_voxels)
        )
        pdb.set_trace()
        # Find distance of x, y, and z from cluster center

        features = torch.from_numpy(features)
        num_voxels[num_voxels == 0] += 1  # Avoids divide by zero
        num_voxels = torch.from_numpy(num_voxels)

        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from beam center
        #f_center = torch.zeros_like(features[:, :, :2])
        #f_center[:, :, 0] = features[:, :, 0] - (coors[:, 0].float().unsqueeze(1) * voxel_size[0]) # X centers
        #f_center[:, :, 1] = features[:, :, 2] - (coors[:, 2].float().unsqueeze(1) * voxel_size[2]) # Z centers

        # Combine together feature decorations
        features_ls = [features, f_cluster]

        if self.augmented:
            features = torch.cat(features_ls, dim=-1)
        elif self.local:
            features = features_ls[1]
        else:
            features = features_ls[0]

        if self.with_distance:
            #points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True) # l2 norm from origin
            points_dist = self.get_dist_feats(features, self.beam_dim) # distance from front/back
            features = torch.cat([features, points_dist], dim=-1)

        # The feature decorations were calculated without regard to whether beam was empty. Need to ensure that
        # empty beams remain set to zeros.
        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        if self.zero_pad:
            features = self.zero_pad_voxels(features)
            num_voxels = self.zero_pad_voxels(num_voxels)

        # logging metrics for voxels
        """sums = features.sum(-1)

        n_dense = len(torch.where(sums != 0.0)[0])
        point_density = n_dense / (sums.shape[0]*sums.shape[1])
        print(f"point-wise density: {point_density}")"""
        pdb.set_trace()
        return features, num_voxels

    @staticmethod
    def fetch_bb_points(pc, bbox, max_points):
        x_min, y_min, z_min, x_max, y_max, z_max = tuple(bbox)
        points = pc[
            (pc[:, 0] > x_min) & (pc[:, 0] < x_max) & (pc[:, 1] > y_min) & (pc[:, 1] < y_max) & (pc[:, 2] > z_min) & (
                        pc[:, 2] < z_max)]
        if isinstance(points, np.ndarray):
            points = torch.Tensor(points)
        num_points = torch.tensor(points.shape[0])
        output = torch.zeros(max_points, points.shape[1])

        idx = min(max_points, points.shape[0])

        output[:idx, :] = points[:idx]

        return output, num_points

    def alternate_beams(self, points, centroids, dimensions, *args, **kwargs):
        """
        This method uses tensor filtering over numba jit.
        Use this method instead of __call__ for debugging. This method is 3x slower than __call__
        :param points:
        :param centroids:
        :param dimensions:
        :param args:
        :param kwargs:
        :return:
        """

        if isinstance(centroids, torch.Tensor):
            centroids = centroids.data.numpy()

        if isinstance(dimensions, torch.Tensor):
            dimensions = dimensions.data.numpy()

        voxel_map = self.xyzwlh2xyzxyz(np.hstack([centroids, dimensions]), self.pc_range, self.epsilon, self.beam_dim)
        feats, num_vxls = zip(*[self.fetch_bb_points(points, dims, self.max_points) for dims in voxel_map])

        features = torch.stack(feats)
        num_voxels = torch.stack(num_vxls)

        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from beam center
        #f_center = torch.zeros_like(features[:, :, :2])
        #f_center[:, :, 0] = features[:, :, 0] - (coors[:, 0].float().unsqueeze(1) * voxel_size[0]) # X centers
        #f_center[:, :, 1] = features[:, :, 2] - (coors[:, 2].float().unsqueeze(1) * voxel_size[2]) # Z centers

        # Combine together feature decorations
        features_ls = [features, f_cluster]
        if self.with_distance:
            #points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True) # l2 norm from origin
            points_dist = self.get_dist_feats(features, self.beam_dim) # distance from front/back
            features_ls.append(points_dist)
        if self.augmented:
            features = torch.cat(features_ls, dim=-1)
        else:
            features = features_ls[0]


        # The feature decorations were calculated without regard to whether beam was empty. Need to ensure that
        # empty beams remain set to zeros.
        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        if self.zero_pad:
            features = self.zero_pad_voxels(features)
            num_voxels = self.zero_pad_voxels(num_voxels)

        # logging metrics for voxels
        """sums = features.sum(-1)

        n_dense = len(torch.where(sums != 0.0)[0])
        point_density = n_dense / (sums.shape[0]*sums.shape[1])
        print(f"point-wise density: {point_density}")"""

        return features, num_voxels

    def zero_pad_voxels(self, tensor):
        """
        Zero pad feature tensor to have max_voxels
        @return: torch.Tensor (self.max_voxels, N, D)
        """

        if tensor.ndim == 3:
            P, N, D = tensor.shape
            padded = torch.zeros(self.max_voxels, N, D)
            padded[:P, :, :] = tensor
        elif tensor.ndim == 2:
            P, D = tensor.shape
            padded = torch.zeros(self.max_voxels, D)
            padded[:P, :] = tensor
        elif tensor.ndim == 1:
            P = tensor.shape[0]
            padded = torch.zeros(self.max_voxels)
            padded[:P] = tensor
        else:
            raise NotImplementedError(f"ndim not supported: {tensor.ndim}")

        # print(f"voxel density: {P / self.max_voxels}" )

        return padded

    @classmethod
    def build(cls, cfg, zero_pad=True):
        beam_features = BeamFeatures(
            max_voxels=cfg["max_voxels"], #TODO: Set to max centroids from imgs
            max_points=cfg["max_points"],
            with_distance=cfg["with_distance"],
            epsilon=cfg["epsilon"],
            zero_pad=zero_pad,
            beam_dim=cfg["beam_dim"]
        )
        return beam_features
