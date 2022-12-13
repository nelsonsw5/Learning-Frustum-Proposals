from enum import Enum
from typing import Optional

from torch.utils.data import Dataset
from dataset_manager.dataset_manager import ItemManager

from transforms import NormalizeShift, YForward2NegZForward, AddNoise, MirrorX #, RandomRotationZ
from features.feature_extraction import *


class DatasetTypes(Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "validation"

class DatasourceTypes(Enum):
    SIMULATED = "SYNTHETIC"
    DISPLAY_AUDIT = "DISPLAY AUDIT"


class PointCloudDataset(Dataset):

    def __init__(
            self,
            dataset: ItemManager,
            mirror_x: Optional[dict] = None,
            add_noise: Optional[dict] = None,
            sample_points: bool = None,
            *args,
            **kwargs
    ):
        self.dataset = dataset
        self.dataset_id = dataset.dataset_name

        self.sample_points = sample_points

        self.transform = NormalizeShift()
        self.swap = YForward2NegZForward()

        # Data Augmentation

        self.add_noise = AddNoise(mean=add_noise["mean"],
                                  std=add_noise["std"],
                                  noise_percentage=add_noise["samples"]) \
            if add_noise and add_noise["apply"] else None

        self.mirror_x = MirrorX(probability=mirror_x["probability"]) if mirror_x and mirror_x["apply"] else None

    def _get_centroids(self, item):
        centroid_ordered_dict = item.centroids.data

        num_centroids = len(centroid_ordered_dict)
        num_classes = len(self.dataset.geomap)

        points = torch.zeros((num_centroids, 3))
        centroid_cls = torch.zeros((num_centroids, 1)).long()

        dim_arr = torch.zeros(num_centroids, 3)

        i = 0

        for _, dta in centroid_ordered_dict.items():            
            points[i, :] = torch.tensor(dta["points"])
            if dta["dims"]:
                dim_arr[i, :] = torch.tensor(dta["dims"])
            i += 1

        points = torch.tensor(np.array(points).astype(np.float32))
        centroid_cls = torch.tensor(np.array(centroid_cls))


        # 
        output = {
            "points": points,
            "centroid_cls": centroid_cls,
            "cluster_counts": item.label.slot_counts,
            "dict": centroid_ordered_dict,
            "dims": dim_arr
        }

        return output

    @staticmethod
    def sample_point_cloud_tensor(points, normals, sem_seg, max_points_per_pointcloud):
        points = points[:max_points_per_pointcloud, :] if points.shape[0] > max_points_per_pointcloud else points
        if sem_seg is not None:
            sem_seg = sem_seg[:max_points_per_pointcloud] if sem_seg.shape[0] > max_points_per_pointcloud \
                else sem_seg
        else:
            sem_seg = None

        if normals is not None:
            normals = normals[:max_points_per_pointcloud] if normals.shape[0] > max_points_per_pointcloud \
                else normals

        return points, normals, sem_seg

    def _verify_valid_source(self, datasource):
        if datasource is None:
            raise ValueError("No provided data_source")

        if datasource == DatasourceTypes.SIMULATED.value or datasource == DatasourceTypes.DISPLAY_AUDIT.value:
            return True, datasource
        else:
            raise NotImplementedError(f"Datasource type '{datasource}' not supported")

    def __getitem__(self, idx):
        batch = {}
        item = self.dataset[idx]

        self._verify_valid_source(item.source)
        verts, normals = item.obj.data

        batch["idx"] = torch.tensor([idx])
        batch["normals"] = normals
        batch["verts"] = verts

        centroids = self._get_centroids(item)

        # TODO semseg
        # if 'semantic_segmentation' in labels:
        #     batch["ss_scores"] = torch.tensor(labels['semantic_segmentation']).unsqueeze(-1)

        # data augmentation: add noise
        batch["verts"] = self.add_noise(batch["verts"]) if self.add_noise else batch["verts"]

        # rescale pointclouds
        batch["verts"] = self.transform.fit_transform(batch["verts"])
        batch["centroids"] = self.transform(centroids["points"])
        if centroids["dims"] is not None:
            centroids["dims"] = self.transform.scale(centroids["dims"])

        if batch["normals"] is not None:
            batch["normals"] = self.transform(batch["normals"])

        # only swap axes for simulated dataset
        if item.source == DatasourceTypes.SIMULATED.value:
            batch["verts"] = self.swap(batch["verts"])
            batch["centroids"] = self.swap(batch["centroids"])
            if batch["normals"] is not None:
                batch["normals"] = self.swap(batch["normals"])
            if centroids["dims"] is not None:
                # don't invert because these are lengths, not points
                # ie, inverting the last dim would result in a negative length
                centroids["dims"] = self.swap(centroids["dims"], invert=False)

        # data augmentation: mirror x
        if self.mirror_x:
            batch["verts"] = self.mirror_x.fit_transform(batch["verts"])
            batch["centroids"] = self.mirror_x(batch["centroids"])
            batch["normals"] = self.mirror_x(batch["normals"]) if batch["normals"] is not None else None

        #TODO requires pt3d
        # data augmentation: rotate
        # if self.rotate_z:
        #     batch["verts"] = self.rotate_z.fit_transform(batch["verts"])
        #     batch["centroids"] = self.rotate_z(batch["centroids"])
        #     batch["normals"] = self.rotate_z(batch["normals"]) if batch["normals"] is not None else None

        batch["fpath"] = item.obj.path

        batch["centroid_cls"] = centroids["centroid_cls"]
        batch["centroid_dict"] = centroids["dict"]
        batch["centroid_dims"] = centroids["dims"]
        batch["labels"] = self._get_labels(item.label)

        return batch


    def _get_labels(self, labels):
        return labels

    def __len__(self):
        return len(self.dataset)



