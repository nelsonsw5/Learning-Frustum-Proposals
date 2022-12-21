from typing import List, Dict

import torch
# from pytorch3d.structures import Pointclouds
# from pytorch3d.ops import knn_points
import pdb

from transforms import MeanShift

def normalize_shift_point_cloud(pc):
    """
    there are codes for such preprocessings in the project (python implementation), it is very intuitive:

    compute mean coordinates (x',y',z'): (x',y',z') = mean(coords(poincloud))
    subtract (x',y',z') from all points: coords(translated_pointcloud) = coords(pointcloud) - (x',y',z')
        ---that is to move the point cloud to the origin
    calculate distances between all points and the origin (0,0,0), and find the maximum: dist_max
    coordinates of all points divided by dist_max:
    coords(final_poincloud) = coords(translated_pointcloud)/dist_max
    """

    means = torch.mean(pc, dim=0)
    translated = pc - means

    origin = torch.zeros(3)
    dist = torch.norm(translated - origin, p=2, dim=1)
    max_dist = torch.max(dist)
    output = translated / max_dist
    return output


def aggregate_nbrs(query_points, nbrs, idx, normals=None, features=None, mean_shift=True):
    """
        args:
            query_points: query points for which to get nieghbors
            must have dims [n_points, n_channels]

            nbrs: nieghbor points for each query point.
            Must have dims [n_points, n_nbrs, n_channels]
            mean_shift: (bool) rescale to local coordinate system; meanshift sub-point clouds

        return:
            pointclouds (Pointclouds): Pointclouds object containing list of aggregated point clouds

    """

    assert query_points.ndim == 2
    assert nbrs.ndim == 3

    P, C = query_points.shape
    _, n_nbrs, _  = nbrs.shape

    if features is not None:
        n_feat_points, d = features.shape
        assert n_feat_points == P

    pc_list = []
    feat_list = []
    normals_list = []

    shift = MeanShift()

    for i in range(P):
        pc_i = torch.zeros((n_nbrs+1, C))
        pc_i[0] = query_points[i]

        if normals is not None:
            normals_i = normals[idx[0][i], :]
            if mean_shift:
                normals_i = shift(normals_i)
            centroid_normal = normals_i.mean(dim=0) # approximate centroid normal with mean
            # concat to get ((k+1), 3) tensor
            normals_list.append(
                torch.cat([centroid_normal.unsqueeze(0), normals_i],dim=0)
            )

        if features is not None:
            f_i = features[i].repeat(n_nbrs+1, 1)
            feat_list.append(f_i)

        for j in range(n_nbrs):
            pc_i[j+1] = nbrs[i, j, :]

        if mean_shift:
            pc_i = shift(pc_i)

        pc_list.append(pc_i)

    sub_point_clouds = Pointclouds(
        pc_list,
        features=feat_list if feat_list else None,
        normals=normals_list if normals_list else None
    )

    return sub_point_clouds

def get_centroid_nbrs_from_pc(centroids, points, n_nbrs, normals=None, features=None, mean_shift=False):
    dist, indx, nbrs = knn_points(centroids.unsqueeze(0), points.unsqueeze(0), K=n_nbrs, return_nn=True)
    agg_centroids = aggregate_nbrs(
        centroids.squeeze(0),
        nbrs.squeeze(0),
        idx=indx,
        normals=normals,
        features=features,
        mean_shift=mean_shift
    )
    return agg_centroids

"""from pytorch3d.ops import knn_points

x = torch.randn(1, 5, 3)
y = torch.randn(1, 100, 3)
f = torch.randn(1, 5, 10)

dist, idx, nbrs =  knn_points(x, y, K=5, return_nn=True)

output = aggregate_nbrs(x.squeeze(0), nbrs.squeeze(0), f.squeeze(0))
stop = 0"""

def get_zero_padded_batch(input: List[torch.Tensor], flatten_last: bool = False):
    max_dim = 0
    batch_size = len(input)

    # make sure all inputs are at least 2D
    batch = []
    for x in input:
        if x.ndim == 1:
            batch.append(
                x.unsqueeze(-1)
            )
        else:
            batch.append(x)


    shape = list(batch[0].shape)
    shape.pop(0)
    dtype = batch[0].dtype

    for x in batch:
        n = x.shape[0]
        if n > max_dim:
            max_dim = n
    shape = [batch_size, max_dim] + shape
    padded = torch.zeros(shape,dtype=dtype)

    for i, x in enumerate(batch):
        n = x.shape[0]
        padded[i,:n, ::] = x

    if flatten_last:
        padded = padded.squeeze(-1)

    return padded



def get_dataset(trn_cfg):
    # Kitti, SunRGBD, 3DBev24K, Coolers
    if trn_cfg[dataset][dir] == "Kitti":
        dataset = Kitti()
    if trn_cfg[dataset][dir] == "SunRGBD":
        dataset = Kitti()
    if trn_cfg[dataset][dir] == "3DBev24K":
        dataset = Kitti()
    if trn_cfg[dataset][dir] == "Coolers":
        dataset = Kitti()
    


    return dataset
