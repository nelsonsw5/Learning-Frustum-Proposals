from tqdm import tqdm
import torch
from typing import Optional

from dataset_manager.dataset_manager import ItemManager, Label

from datasets.point_cloud_data import PointCloudDataset
from transforms import MaxNormalizer, NormalizeShift, NegZForward2YForward, MirrorX, ShufflePoints, GaussianResize

from features.points_to_beams import BeamFeatures
from features.points_to_frustums import FrustumFeatures
from models.model_utils import NeonetTypes
import numpy as np
# from features.bev import *




class BaseDensityDataset(PointCloudDataset):

    wandb_point_limit = 20000

    def __init__(
            self,
            dataset: ItemManager,
            max_val: Optional[int] = None,
            rescale_outputs: bool = False,
            sample_points: Optional[bool] = None,
            shuffle: bool = False,
            seg_filter: bool = False,
            classification: bool = False,
            mirror_x: Optional[dict] = None,
            add_noise: Optional[dict] = None,
            resize: Optional[dict] = None,
            *args,
            **kwargs
    ):
        super(BaseDensityDataset, self).__init__(
            dataset,
            rescale_outputs=rescale_outputs,
            sample_points=sample_points,
            read_sem_seg=seg_filter,
            mirror_x=mirror_x,
            add_noise=add_noise,
            *args,
            **kwargs
        )
        self.label_norm = MaxNormalizer(max=max_val, scale=rescale_outputs)
        self.shuffle = ShufflePoints()
        self.do_shuffle = shuffle
        self.seg_filter = seg_filter
        self.classification = classification

        if resize:
            self.resize = GaussianResize(
                mu=resize["mean"],
                sig=resize["std"],
                samples=resize["samples"],
                broadcast=True
            )
        else:
            self.resize = None

    def get_n_geo_types(self):
        geoidx = set(self.dataset.geomap.values())
        return len(geoidx)

    def get_pc_feats(self, points):
        """

        @param points: (torch.Tensor) n x 3 dims
        @return: feats_rgb (torch.Tensor)
        """

        feats = points[:, :, :3]
        n, d, c = feats.shape  # centroids, dimension, channels
        feats = torch.flatten(feats, start_dim=0, end_dim=1)

        f_colors = []
        for i in range(n):
            color = np.random.randint(0, 255, size=3).reshape(1, 3)
            rgb_i = np.repeat(color, d, axis=0)
            f_colors.append(rgb_i)
        f_colors = np.concatenate(f_colors)
        feats_rgb = np.concatenate([feats, f_colors], axis=1)

        return feats_rgb

    @staticmethod
    def binarize_seg_scores(seg, tol=0.5):
        if seg is not None:
            return torch.where(seg > 0.5, 1, 0)
        else:
            return seg

    def get_points_by_seg(self, points, seg_feats):
        if self.seg_filter and seg_feats is not None:
            return points[seg_feats.squeeze(-1).bool()]
        else:
            return points


    def get_wandb_plot(self, wandb_run, batch, y, y_hat=None, eval_dict=None):
        swap = NegZForward2YForward()
        flip = MirrorX(probability=1.0)

        if y.ndim == 1:
            # reshape tensor if single dimension is give
            y = y.unsqueeze(-1)

        num_targets = batch["num_targets"].data.numpy().flatten().tolist()
        num_verts = batch["num_verts"].data.numpy().flatten().tolist()
        batch_size = len(num_targets)

        imgs = batch.get("img", None)

        idx = 0
        verts_idx = 0
        for i, n in enumerate(num_targets):
            names = batch['names'][idx:(n + idx)]

            # unswap axes for viz
            points = flip.fit_transform(swap(batch["verts"][verts_idx:(verts_idx + num_verts[i])]))
            centroids = flip(swap(batch["centroids"][idx:(n+idx), ::]))
            dims = swap(batch["dims"][idx:(n+idx), ::])

            # handle 4D input
            if batch["points"].ndim > 3:
                keep_idx = torch.where(y[i] > 0)
                feats = swap(batch["points"][i,:, :, :3][keep_idx])
                y_i = y[i][keep_idx]
                gt_labs = y_i.data.numpy().flatten().tolist()

                if y_hat is not None:
                    if y_hat.ndim == 1: # if batch size = 1, need to reshape
                        y_hat = y_hat.unsqueeze(0)
                    y_hat_i = y_hat[i][keep_idx]
                    pred_labs = y_hat_i.data.numpy().flatten().tolist()

            # handle 3d input
            else:
                feats = swap(batch["points"][idx:(n+idx), :, :3])
                gt_labs = y[idx:(n + idx), :].data.numpy().flatten().tolist()
                if y_hat is not None:
                    pred_labs = np.round(y_hat[idx:(n + idx)].data.numpy().flatten().tolist(), 1)

            feats[:, :, 0] = -1*feats[:, :, 0]
            feats_rgb = self.get_pc_feats(feats)


            if y_hat is not None:
                labels = [f"{names[j]}\ngt: {gt_labs[j]}\npred: {pred_labs[j]}" for j in range(len(gt_labs))]
            else:
                labels = [f"{names[j]}\ngt: {gt_labs[j]}" for j in range(len(gt_labs))]



            boxes = wandb_run.get_bb_dict(centroids, dims, labels=labels)
            points = points.data.numpy()
            rgb = wandb_run.get_rgb_point_heatmap(points)

            # Fetch points (with associated colors) for logging in W&B later
            points_rgb = np.array([[p[0], p[1], p[2], c[0], c[1], c[2]] for p, c in zip(points, rgb)])

            # log point clouds
            pc_dict = wandb_run.get_point_cloud_log(
                points=points_rgb,
                boxes=np.array(boxes),
                key="Point Cloud"
            )

            pc_feats_dict = wandb_run.get_point_cloud_log(
                points=feats_rgb,
                boxes=np.array(boxes),
                key="Point Features"
            )

            dict_list = [pc_dict, pc_feats_dict]

            # log images
            if imgs:
                img_paths = [im.get_path() for im in imgs[i]]
                img_dict = wandb_run.get_img_log(img_paths)
                dict_list.append(img_dict)

            if eval_dict:
                dict_list.append(eval_dict)


            # get semantic seg rgb vals
            ss = batch.get('ss_scores', None)
            if ss is not None:
                ss = ss[verts_idx:(verts_idx + num_verts[i])]
                seg_rgb = wandb_run.get_point_seg_rgb(ss)
                point_seg_rgb = np.array([[p[0], p[1], p[2], c[0], c[1], c[2]] for p, c in zip(points, seg_rgb)])

                seg_feats_dict = wandb_run.get_point_cloud_log(
                    points=point_seg_rgb,
                    boxes=np.array(boxes),
                    key="Point Segmentation"
                )

                dict_list.append(seg_feats_dict)

            log_dict = {}
            for d in dict_list:
                for k, v in d.items():
                    log_dict[k] = v

            wandb_run.log(log_dict)

            idx += n
            verts_idx += num_verts[i]

    def get_img_list(self, item):
        return item.images

    def shuffle_points(self, verts, sem_seg):
        if self.do_shuffle:
            verts = self.shuffle.fit_transform(verts)

        if sem_seg is not None and self.do_shuffle:
            sem_seg = self.shuffle(sem_seg)

        return verts, sem_seg


    @staticmethod
    def _get_geometry_map(labels):
        print("Indexing geometric types")
        geometry_map = {}
        i = 0
        for id, scene_dict in tqdm(labels.items(), total=len(labels)):
            for c_id, cluster_dict in scene_dict["slot_counts"].items():
                if cluster_dict["object_type"] not in geometry_map:
                    geometry_map[cluster_dict["object_type"]] = i
                    i += 1
        return geometry_map

    def _get_labels(self, label: Label):
        y = label.slot_counts

        if self.classification:
            y -= 1
            y = y.long()
        else:
            y = y.float()
            y = self.label_norm(y)

        return y

    def _get_geometric_type(self, centroid_dict):
        n_geos = self.get_n_geo_types()
        n_centroids = len(centroid_dict)
        one_hot = torch.zeros((n_centroids, n_geos))
        names = []

        i = 0
        # ordered dict
        for geoshape, dta in centroid_dict.items():
            shp_type = geoshape.split("_")[0]
            j = self.dataset.geomap[shp_type]
            one_hot[i, j] = 1
            names.append(geoshape)
            i += 1

        return one_hot, names

class PointBeamDensityDataset(BaseDensityDataset):

    def __init__(
            self,
            dataset: ItemManager,
            max_points: int,
            eps: float,
            augmented: bool,
            max_val: int,
            rescale_outputs: Optional[bool] = False,
            sample_points: Optional[bool] = None,
            seg_filter: bool = False,
            shuffle: bool =False,
            distance: bool = False,
            mean_shift: bool = False,
            classification: bool = False,
            mirror_x: Optional[dict] = None,
            add_noise: Optional[dict] = None,
            resize: Optional[dict] = None,
            *args,
            **kwargs
    ):
        BaseDensityDataset.__init__(self,
            dataset=dataset,
            max_val=max_val,
            rescale_outputs=rescale_outputs,
            sample_points=sample_points,
            shuffle=shuffle,
            seg_filter=seg_filter,
            classification=classification,
            mirror_x=mirror_x,
            add_noise=add_noise,
            resize=resize,
            *args,
            **kwargs
        )
        self.max_points=max_points
        self.eps=eps
        self.augmented=augmented
        self.shuffle = ShufflePoints()
        self.distance = distance
        self.mean_shift = mean_shift

    def get_upcs_from_centroid_dict(self, centroid_dict):
        values_list = list(centroid_dict.values())
        upcs_list = list(map(lambda x: x.get("class_label", None), values_list))
        return upcs_list


    def __getitem__(self, idx):
        batch = PointCloudDataset.__getitem__(self, idx)
        item = self.dataset[idx]
        y = batch["labels"]

        verts = batch["verts"]
        # sem_seg = batch.get('ss_scores', None)

        verts, _ = self.shuffle_points(verts, None)

        if self.sample_points:
            verts, batch["normals"], _ = self.sample_point_cloud_tensor(
                verts,
                batch["normals"],
                None,
                self.sample_points)

        beam_feats = BeamFeatures(
            beam_dim=2, # assumed to be in -Z forward (App) space
            max_voxels=len(batch["centroids"]),
            max_points=self.max_points,
            epsilon=self.eps,
            augmented=self.augmented,
            with_distance=self.distance,
            local=self.mean_shift,
            zero_pad=True, # TODO: Revisit this; some centroids are missing points in scenes
            resize=self.resize
        )

        feats, _ = beam_feats(
            points=verts,
            centroids=batch["centroids"],
            dimensions=batch["centroid_dims"]
        )


        geo_feats, names = self._get_geometric_type(batch["centroid_dict"])

        x = {
            "points": feats,
            "object_geo_type":geo_feats,
            "names": names,
            "num_targets": torch.tensor(batch["centroids"].shape[0]).unsqueeze(0),
            "cls": batch["centroid_cls"],
            "dims": batch["centroid_dims"],
            "centroids": batch["centroids"],
            "verts": verts[:self.wandb_point_limit, :], # truncate (shuffled) points for wandb viz
            "num_verts": torch.tensor(min(verts.shape[0], self.wandb_point_limit)).unsqueeze(0),
            "idx": torch.full((len(batch["centroids"]),1),batch["idx"].item()),
            "upcs": self.get_upcs_from_centroid_dict(batch["centroid_dict"])
        }

        img = self.get_img_list(item)
        if len(img) > 0 and img[0]:
            x.update({"img": img})

        return x, y




def get_dataset(item_manager, job_cfg, model_cfg, max_val=None, max_norm=False):
    """

    @param item_manager: (ItemManager) Dataset manager instance
    @param job_cfg: (dict) training job config
    @param model_cfg: (dict) model config
    @param max_norm: (bool) whether or not to normalize the outputs
    @return: DensityPointCloudDataset
    """

    data_aug = job_cfg["data_augmentation"]

    if model_cfg["point_features"]["type"] == "point_beams":
        dta = PointBeamDensityDataset(
            dataset=item_manager,
            max_val=max_val,
            rescale_outputs=max_norm,
            eps=model_cfg["point_features"]["epsilon"],
            max_points=model_cfg["point_features"]["max_points"],
            augmented=model_cfg["point_features"]["augmented"],
            seg_filter=model_cfg['point_features']['seg_filter'],
            distance=model_cfg['point_features']['distance'],
            shuffle=job_cfg["dataset"]["shuffle"],
            mean_shift=model_cfg["point_features"]["mean_shift"],
            sample_points=job_cfg["dataset"]["sample_points"],
            classification=True if model_cfg["head"]["type"] == NeonetTypes.CLASSIFIER.value else False,
            add_noise=data_aug["add_noise"] if data_aug["add_noise"]["apply"] else None,
            mirror_x=data_aug["mirror_x"] if data_aug["mirror_x"]["apply"] else None,
            resize=data_aug["resize"] if data_aug["resize"]["apply"] else None
        )
    else:
        raise NotImplementedError(f"point_features type {model_cfg['type']}, not supported")

    return dta
