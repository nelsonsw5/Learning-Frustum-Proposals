import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import argparse


from torch.utils.data import DataLoader

from models.model_utils import get_max_normalizer, load_model_chkp
from models.neonet import build_from_yaml, get_loss_fn
from evaluators.density_evaluator import EVALUATORS

from datasets.collate_func import get_collate_fn
from datasets.density_point_cloud_data import get_dataset
from dataset_manager.dataset_manager import DatasetManager

from utils import get_yaml


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--cfg', type=str, default="./cfg_neonet_eval.yaml", help='job yaml config file')
    return parser.parse_args()

def main(args):
        job_cfg = get_yaml(args.cfg)
        model_cfg = get_yaml(job_cfg["model"]["cfg"])
        task = model_cfg["head"]["type"]

        dataset_manager = DatasetManager(job_cfg["dataset"]["dir"])

        print("Total items:", len(dataset_manager))

        print("Filtering All Ones")
        dataset_manager.filter_all_ones()

        print("Filtering Scenes with 0 counts")
        dataset_manager.filter_scenes_with_zero_counts()

        print("Filtering point clouds with missing data")
        dataset_manager.filter_missing_data()

        print("Total items after filtering:", len(dataset_manager))

        # get max normalizer from training data
        max_val = get_max_normalizer(job_cfg["model"]["normalizer"])

        test = get_dataset(
            dataset_manager.test,
            job_cfg,
            model_cfg,
            max_val=max_val,
            max_norm=False
        )
        test_loader = DataLoader(
            test,
            batch_size=job_cfg["optimization"]["batch_size"],
            shuffle=False,
            collate_fn=get_collate_fn(model_cfg["point_encoder"]["type"])

        )

        model, model_cfg = build_from_yaml(
            fpath_or_dict=model_cfg,
            num_geo_types=test.get_n_geo_types(),
            use_cuda=job_cfg["optimization"]["cuda"],
            max_val=max_val
        )


        model = load_model_chkp(
            model,
            job_cfg["model"]["chkp"],
            job_cfg["optimization"]["cuda"],
            strict=True
        )

        evaluator = EVALUATORS[task](
            model=model,
            test_dataloader=test_loader,
            cuda=job_cfg["optimization"]["cuda"],
            gen_plots=job_cfg["metadata"]["plot"],
            output_dir="./runs/",
            log_wandb=job_cfg["metadata"]["wandb_log"],
            wandb_prj=job_cfg["metadata"]["wandb_project"],
            wandb_entity=job_cfg["metadata"]["wandb_entity"],
            wandb_group=job_cfg["metadata"]["wandb_group"],
            wandb_run_name=job_cfg["metadata"]["wandb_run"],
            limit=job_cfg["dataset"].get("limit", None),
            log=[job_cfg, model_cfg],
            max_val=max_val,
            loss_fn=get_loss_fn(task),
        )
        evaluator.main()
        if job_cfg["diff"]["apply"]:
            evaluator.diff_eval(num_cands=job_cfg["diff"]["k"])

        if job_cfg["metadata"]["gcs_log"]:
            evaluator.log_gcs_chkp(
                chkp_dir=os.path.split(job_cfg["model"]["chkp"])[0],
                bucket=job_cfg["metadata"]["gcs_bucket_path"],
                auth=job_cfg["metadata"]["gcs_auth"],
                overwrite=False
            )


if __name__ == "__main__":
        args = parse_args()
        main(args)