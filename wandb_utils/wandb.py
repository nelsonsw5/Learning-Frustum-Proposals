import pandas as pd
import torch
from matplotlib import colors as colors

import wandb
from datetime import datetime
import numpy as np
import os
import pdb

from torchvision.utils import make_grid
from PIL import Image

from bound_box import ThreeDimBoundBox

class WandB:
    def __init__(
        self, 
        project,
        config={},
        entity="prj-research",
        enabled=False, 
        group="Default",
        job_type=None,
        name=None,
        log_objects=True
    ):
        """Weights and Biases wrapper class to contain all logic dependant on the wandb python library

        Args:
            project (str): Name of WandB project (3d-count-inference, etc.)
            config (dict, optional): Dictionary containing WandB run metadata. Defaults to {}.
            entity (str, optional): Name of WandB entity. Defaults to "prj-research".
            enabled (bool, optional): If set to false, no wandb logic will execute.
                                      This way no conditional logic needs to happen outside of this file.
                                      Defaults to False.
            group (str, optional): Name of WandB group (for organizing runs within a project). Defaults to "Default".
            job_type (str, optional): Used for more organization (Train/Eval, Simulated/Real, etc.). Defaults to None.
            name (str, optional): Give a custom name to the run. Defaults to None, which gets set as the current date and time.
            log_objects (bool, optional): If set to False, no objects will be uploaded
                                          (e.g. if you want a run logged, but don't want to waste storage on point clouds you don't need).
                                          Defaults to True.
        """
        self.enabled = enabled

        if enabled:
            if name is None:
                name = str(datetime.now().strftime("%m-%d-%Y_%H:%M:%S"))

            self.run = wandb.init(project=project, entity=entity, group=group, job_type=job_type, name=name, config=config)
            self.run.config.update(config)
            self.config = self.run.config
            self.log_objects = log_objects

    def get_name(self):
        return self.run.name if self.enabled else None

    def get_log_dir(self, chkp_dir):
        if self.enabled:
            if self.run.name:
                log_dir = os.path.join(chkp_dir, self.run.name)
            else:
                log_dir = os.path.join(chkp_dir, self.run.id)
            return log_dir
        return "./chkp/"

    def update_name(self, name):
        if self.enabled:
            self.run.name = name

    def update_config(self, config_vars):
        if self.enabled:
            self.config.update(config_vars)

    def watch_model(self, model):
        if self.enabled:
            wandb.watch(model)

    def log(self, log):
        if self.enabled:
            self.run.log(log)

    def log_file_reference(self, reference, name, type):
        if self.enabled:
            artifact = wandb.Artifact(name, type=type)
            artifact.add_reference('gs://' + reference)
            self.run.log_artifact(artifact)

    def log_file(self, file_path, name, type):
        if self.enabled and self.log_objects:
            artifact = wandb.Artifact(name, type=type)
            artifact.add_file(file_path)
            self.run.log_artifact(artifact)

    def log_dir(self, dir, name, type):
        if self.enabled and self.log_objects:
            artifact = wandb.Artifact(name, type=type)
            artifact.add_dir(dir)
            self.run.log_artifact(artifact)

    def log_image(self, name, image_path):
        if self.enabled and self.log_objects:
            wandb.log({name: wandb.Image(image_path)})

    def get_img_log(self, fpath_or_list, key="Key Frames", caption=None, itrid=0):
        """

        @param fpath_or_list: {str, list}
            - if (str): expects a filepath
            - if (list): expects a list of filepaths
        @param key: (str) name of image object, key for dict
        @return:
        """
        if isinstance(fpath_or_list, list):
            img_list = []
            for path_i in fpath_or_list:
                img_i = Image.open(path_i)
                img_i = img_i.resize((640, 480))
                img_i = np.array(img_i)
                img_i = torch.Tensor(np.transpose(img_i, [2, 0, 1]))
                img_list.append(img_i)

            img_arr = make_grid(img_list, nrow=2)
            # [C, H, W] == > [H, W, C]
            img_arr = torch.transpose(img_arr, 0, 2)
            img_arr = torch.transpose(img_arr, 0, 1)
            img_arr = img_arr.data.numpy()

            image = wandb.Image(img_arr, caption=caption)
        elif isinstance(fpath_or_list, torch.Tensor):
            fpath_or_list = make_grid(fpath_or_list, nrow=8)
            image = wandb.Image(fpath_or_list, caption=caption)

        else:
            image = wandb.Image(fpath_or_list, caption=caption)
        img_dict = {key: image}

        return img_dict

    def get_point_cloud_log(self, points, boxes=None, key="Point Cloud"):
        """

        @param points: (ndarray) np array of point clouds
        @param boxes: (ndarray) np array of boudning boxes (optional)
        @param key: (str) name of point cloud object, key for dict

        @return:
        """
        object_def = {
            "type": "lidar/beta",
            "points": points
        }
        if boxes is not None:
            object_def["boxes"] = boxes
        # pdb.set_trace()
        pc_dict = {
            key: wandb.Object3D(object_def)
        }

        return pc_dict if (self.enabled and self.log_objects) else {}

    def log_point_cloud(self, points):
        if self.enabled and self.log_objects:
            self.log(self.get_point_count_log(points))
        
    def add_eval_tables(
        self,
        test_eval_dict,
        per_display_results,
        per_class_results,
        per_object_results,
        metric_types,
        upc_to_object_type=None,
    ):
        if self.enabled:
            print("Adding evaluation result tables to Weights and Biases")
            row = []
            for metric_type in metric_types:
                row.append(test_eval_dict[metric_type].item())
            table = wandb.Table(
                columns=[metric_type for metric_type in metric_types],
                data=[row]
            )
            name = "Overall metric summaries"
            self.log({name: table})

            for results, result_type in [(per_display_results, "display"), (per_class_results, "class"), (per_object_results, "object")]:
                table_data = []
                for instance_type, metrics in results.items():
                    num_instances = sum([len(metrics[metric_type]) for metric_type in metric_types])
                    row = [instance_type, num_instances]
                    for metric_type in metric_types:
                        row.append(np.mean([instance["metric"] for instance in metrics[metric_type]]))
                    table_data.append(row)
                table = wandb.Table(
                    columns=[f"{result_type} type", f"number of instances"] + [metric_type for metric_type in metric_types],
                    data=table_data,
                )
                name = f"Summary results by {result_type}"
                self.log({name: table})

                table_data = []
                for instance_type, metrics in results.items():
                    for instance_metrics in zip(*[metrics[metric_type] for metric_type in metric_types]):
                        num_objects = sum([instance["num_objects"] for instance in instance_metrics])
                        row = [instance_type, num_objects]
                        if result_type == "class":
                            if upc_to_object_type:
                                row.append(upc_to_object_type[instance_type])
                            else:
                                row.append("Unknown")
                        for metric in instance_metrics:
                            row.append(metric["metric"])
                        table_data.append(row)
                metadata = [f"{result_type} type", f"Number of objects"]
                if result_type == "class":
                    metadata[0] = "UPC"
                    metadata.append("object type")
                columns = metadata + [metric_type for metric_type in metric_types]
                table = wandb.Table(
                    columns=columns,
                    data=table_data,
                )
                name = f"Instance results by {result_type}"
                self.log({name: table})

    def finish(self):
        if self.enabled:
            wandb.finish()


    @staticmethod
    def get_bb_dict(centroids=None, dims=None, colors=None, labels=None, boxes=None):
        """
        helper function to get wandb bb format
        @param centroids: (ndarray or Tensor): (n x 3), n = number of centroids
        @param dims: (ndarray or Tensor): (n x 3), n = number of centroids
        @param colors: (list): list of n colors, one for each bounding box
        @param colors: (list): list of n labels, one for each bounding box

        @return: (ndarray) array of dicts compatible with
        """
        if boxes is not None:
            return np.array(boxes)
        if isinstance(centroids, torch.Tensor):
            centroids = centroids.data.numpy()
        if isinstance(dims, torch.Tensor):
            dims = dims.data.numpy()
        boxes = []
        for i, c in enumerate(centroids):
            # pdb.set_trace()
            bb = ThreeDimBoundBox(
                centroids=c,
                dims=dims[i, :]
            )
            # pdb.set_trace()
            box_arr = bb.get_corners().transpose()

            # expects corners as list of tuples:
            # [
            #   (x1, y1, z1),
            #   (x2, y2, z2),
            # ....
            # ]
            box_dict = {
                    "corners": list(zip(*box_arr.tolist())),
                    "color": colors[i] if colors else [0, 255, 0]  # green boxes if none given
                }
            if labels:
                box_dict["label"] = labels[i]
            boxes.append(box_dict)

        return np.array(boxes)

    @staticmethod
    def get_point_seg_rgb(points):
        """
        get red/green colors for semantic segmentation visualization
        @param points: (ndarray) point cloud with {n x 1} dims
        @return:
        """

        colors = np.array([
            [255, 0, 0],
            [0, 255, 0]
        ])

        if isinstance(points, torch.Tensor):
            points = points.squeeze(-1).data.numpy()

        rgb = colors[points]
        return rgb


    @staticmethod
    def get_rgb_point_heatmap(points):
        """
        get pretty colors proportional to distance from center point
        @param points: (ndarray) point cloud with {n x 3} dims
        @return:
        """

        c = np.array([255, 158, 0]) / 255.0
        df_tmp = pd.DataFrame(points, columns=["x", "y", "z"])
        df_tmp["norm"] = np.sqrt(np.power(df_tmp[["x", "y", "z"]].values, 2).sum(axis=1))
        min_val = df_tmp["norm"].min()
        max_val = df_tmp["norm"].max()

        df_tmp["norm"] = (df_tmp["norm"] - min_val) / (max_val - min_val)
        rgb = np.array([colors.hsv_to_rgb([n, 0.4, 0.5]) for n in df_tmp["norm"]]) * 255.0
        return rgb