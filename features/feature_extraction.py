import torch
import numpy as np
# from sklearn.neighbors import NearestNeighbors


class Features(object):
    @classmethod
    def compose(cls, extractors, batch_dict, **kwargs):
        x = []
        for e in extractors:
            x.append(e(batch_dict, **kwargs))
        x = torch.cat(x, dim=-1)
        return x


class FrontFacingFeatures(object):
    def __call__(self, batch_dict, **kwargs):
        return batch_dict["centroid_counts"]

class CentroidFeatures(object):
    def __init__(self, max_from_img=40):
        self.max_from_img = max_from_img
    def __call__(self, batch_dict, **kwargs):
        x = torch.zeros(self.max_from_img, 3)
        for i, c in enumerate(batch_dict["centroids"]):
            x[i] = c
        x = torch.flatten(x)
        return x

class CentroidMtxFeatures(object):
    def __init__(self, max_from_img, n_features=3):
        self.max_from_img = max_from_img
        self.n_features=n_features
    def __call__(self, centroid_arr, **kwargs):
        x = torch.zeros(self.max_from_img, self.n_features)
        for i, c in enumerate(centroid_arr):
            x[i] = c
        return x



class GaussianDensity(object):

    def __init__(self, sigma, n_nbrs, max_from_img=40):
        self.sigma = sigma
        self.n_nbrs = n_nbrs
        self.max_from_img = max_from_img

    def gaussian_window(self, d):
        return np.exp(-(d/self.sigma)**2)

    def weighted_avg(self, rho, weights):
        top = np.dot(rho, weights)
        bottom = np.sum(weights)
        return top / bottom


    def __call__(self, points, **kwargs):
        points = points["centroids"]
        densities = kwargs["voxels"].densities().flatten().data.numpy()
        coors = kwargs["voxels"].get_coord_grid(world_coordinates=True).reshape(-1, 3)
        n_vox = len(densities)

        features = np.zeros(self.max_from_img)

        nbrs = NearestNeighbors(n_neighbors=self.n_nbrs)
        nbrs.fit(coors)


        for i, p_i in enumerate(points):
            weights = np.zeros(self.n_nbrs)
            nbr_densities = np.zeros(self.n_nbrs)
            dist, idx = nbrs.kneighbors(p_i.reshape(1, -1))
            for j in range(self.n_nbrs):
                rho = densities[idx[0, j]]
                w_ij = self.gaussian_window(dist[0, j])
                weights[j] = w_ij
                nbr_densities[j] = rho

            f_i = self.weighted_avg(nbr_densities, weights)
            features[i] = f_i

        return torch.from_numpy(features.astype(np.float32))


class GaussianDensityHistogram(object):

    def __init__(self, sigma, n_nbrs, n_bins=10, max_from_img=40):
        self.sigma=sigma
        self.n_bins=n_bins
        self.n_nbrs = n_nbrs
        self.max_from_img = max_from_img

    def gaussian_window(self, d):
        return np.exp(-(d/self.sigma)**2)

    def __call__(self, points, **kwargs):
        points = points["centroids"]
        densities = kwargs["voxels"].densities().flatten().data.numpy()
        d_range = (np.min(densities), np.max(densities))
        bins = np.linspace(d_range[0], d_range[1], self.n_bins)
        coors = kwargs["voxels"].get_coord_grid(world_coordinates=True).reshape(-1, 3)
        n_vox = len(densities)

        nbrs = NearestNeighbors(n_neighbors=self.n_nbrs)
        nbrs.fit(coors)

        features = np.zeros((self.max_from_img, self.n_bins))

        for i, p_i in enumerate(points):
            hist = np.zeros(self.n_bins)
            dist, idx = nbrs.kneighbors(p_i.reshape(1, -1))

            for j in range(self.n_nbrs):
                rho_j = densities[idx[0, j]]

                w_ij = self.gaussian_window(dist[0, j])
                hist_idx = np.digitize(rho_j, bins)
                hist[hist_idx-1] += w_ij

            features[i, :] = hist / hist.sum()
        return torch.from_numpy(features.astype(np.float32)).flatten()


class GaussianDensityGradientHist(object):

    def __init__(self, sigma, n_nbrs, n_bins=10, gradient_min=0.0, gradient_max=4.0, max_from_img=40):
        self.sigma=sigma
        self.n_bins=n_bins
        self.gradient_min=gradient_min
        self.graient_max=gradient_max
        self.n_nbrs = n_nbrs
        self.max_from_img = max_from_img

    def gaussian_window(self, d):
        return np.exp(-(d/self.sigma)**2)

    def __call__(self, points, **kwargs):
        points = points["centroids"]
        densities = kwargs['voxels'].densities().flatten().data.numpy()
        gradients = np.gradient(densities)
        bins = np.linspace(self.gradient_min, self.graient_max, self.n_bins)

        coors = kwargs['voxels'].get_coord_grid(world_coordinates=True).reshape(-1, 3)
        n_vox = len(densities)
        features = np.zeros((self.max_from_img, self.n_bins))

        nbrs = NearestNeighbors(n_neighbors=self.n_nbrs)
        nbrs.fit(coors)

        for i, p_i in enumerate(points):
            hist = np.zeros(self.n_bins)
            dist, idx = nbrs.kneighbors(p_i.reshape(1, -1))

            for j in range(self.n_nbrs):
                grad_j = gradients[idx[0, j]]
                w_ij = self.gaussian_window(dist[0, j])
                hist_idx = np.digitize(grad_j, bins)
                hist[hist_idx-1] += w_ij

            features[i, :] = hist / hist.sum()
        return torch.from_numpy(features.astype(np.float32)).flatten()

