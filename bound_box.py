from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pdb

TEST_COORS = [
    [0.4156847596168518, -0.06592566519975662, 0.6647587418556213],
    [0.4156847596168518, -0.06592566519975662, 0.9921023845672607],
    [0.4156847596168518, 0.06592556834220886, 0.9921023845672607],
    [0.4156847596168518, 0.06592556834220886, 0.6647587418556213],
    [0.5475358963012695, -0.06592566519975662, 0.6647587418556213],
    [0.5475358963012695, -0.06592566519975662, 0.9921023845672607],
    [0.5475358963012695, 0.06592556834220886, 0.9921023845672607],
    [0.5475358963012695, 0.06592556834220886, 0.6647587418556213]
]


class BoundBox(object):

    def __init__(self, points):
        self.points = np.array(points)
        self.faces = self._get_faces(self.points)

    def _get_faces(self, points):
        adj_list = set()

        for i, p_i in enumerate(points):
            for j, p_j in enumerate(points):
                if np.sum(p_i == p_j) == 2:
                    adj_list.add(tuple(sorted((i, j))))

        return adj_list


    def plot_points(self):
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]

        ax.scatter3D(x, y, z, c=z, cmap='hsv')
        plt.savefig("tmp.png")

    def plot_points_and_lines(self, ax=None):
        interpolate = 100

        if not ax:
            fig = plt.figure()
            ax = plt.axes(projection="3d")


        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]

        ax.scatter3D(x, z, -y, c=z, cmap='hsv')

        for f in self.faces:
            p_1 = self.points[f[0], :]
            p_2 = self.points[f[1], :]
            vary_idx = np.where(p_1 != p_2)[0][0]

            if vary_idx == 0:
                x_line = np.linspace(p_1[vary_idx], p_2[vary_idx], interpolate)

                y_line = np.ones(interpolate) * p_1[1]
                z_line = np.ones(interpolate) * p_2[2]
                #ax.plot3D(x_line, y_line, z_line, 'red')

            if vary_idx == 1:
                x_line = np.ones(interpolate) * p_1[0]
                y_line = np.linspace(p_1[vary_idx], p_2[vary_idx], interpolate)
                z_line = np.ones(interpolate) * p_2[2]
                #ax.plot3D(x_line, y_line, z_line, 'red')

            if vary_idx == 2:
                z_line = np.linspace(p_1[vary_idx], p_2[vary_idx], interpolate)
                x_line = np.ones(interpolate) * p_1[0]
                y_line = np.ones(interpolate) * p_2[1]

            ax.plot3D(x_line, z_line, -y_line, 'red')

        return ax



class ThreeDimBoundBox(object):
    def __init__(self, centroids=None, dims=None, corners=None):
        """

        @param centroids: (array-like) bounding box centroids (1 x 3)
        @param dims: (array-like) dimensions of bounding boxes (1x3)  --> (w, d, h)
        @param points: 8 points defining bounding box

        Either centriods AND dims OR points should be provided

        """
        self.centroid=centroids
        self.dims=dims
        self.corners=corners

        assert (self.corners is not None) or (self.centroid is not None and self.dims is not None)


    def get_midpoint_dims(self):
        """
        maps from 8 point array to centroid, h,w,d representation
        @return: centroids, dimensions
        """

        """if self.centroid is None:
            self.centroid = np.mean(self.corners, axis=0)

        if self.dims is None:
            max = np.max(self.corners, axis=0)
            min = np.min(self.corners, axis=0)

            self.dims = (max - min)

        return self.centroid, self.dims"""

        raise NotImplementedError


    def get_corners(self):
        if self.corners is not None:
            return self.corners
        else:

            x, y, z = self.centroid
            w, d, h = self.dims
            
            # get plane in x, y
            plane = [
                [x + w/2, y + d/2],
                [x + w/2, y - d/2],
                [x - w / 2, y + d / 2],
                [x - w / 2, y - d / 2]
            ]
            
            cube = []
            for point in plane:
                cube.append(
                    [point[0], point[1], z + h/2],
                )
                cube.append(
                    [point[0], point[1], z - h / 2],
                )
            self.corners = np.array(cube)
            return self.corners


if __name__ == "__main__":
    box = ThreeDimBoundBox(corners=TEST_COORS)

    c, d = box.get_midpoint_dims()

    box_2 = ThreeDimBoundBox(centroids=c, dims=d)

    coors = box_2.get_corners()

    print(coors)
    print(coors.shape)