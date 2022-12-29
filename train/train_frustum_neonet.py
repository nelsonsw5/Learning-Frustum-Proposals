import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import pdb


import argparse
from torch.utils.data import DataLoader

from datasets.collate_func import get_collate_fn
from datasets.dataset_utils import get_dataset

from models.frustum_neonet import build_from_yaml, get_loss_fn
from models.model_utils import load_model_chkp, NeonetTypes

from trainers.learned_frustum_trainer import TRAINERS
from train.train_utils import setup

from utils import get_yaml
from datasets.kitti.kitti_dataset import KittiDataset
# from datasets.sunrgbd.sunrgbd_dataset import S



def main(args):
    trn_cfg = get_yaml(args.trn_cfg)
    # optional model cfg from command line:
    if args.model_cfg is not None:
        print(f"override with command line path: {args.model_cfg}")
        model_cfg = get_yaml(args.model_cfg)
    else:
        model_cfg = get_yaml(trn_cfg["model"]["cfg"])

    max_val = 15
    task = model_cfg["head"]["type"]
    trn = KittiDataset(data_path='/Users/stephennelson/Projects/Data/Kitti/')

    trn_loader = DataLoader(
        trn,
        batch_size=trn_cfg["optimization"]["batch_size"],
        shuffle=True
    )

    test = KittiDataset(data_path='/Users/stephennelson/Projects/Data/Kitti/',test=True)

    test_loader = DataLoader(
        test,
        batch_size=trn_cfg["optimization"]["test_batch_size"],
        shuffle=False
    )

    model, model_cfg = build_from_yaml(
        fpath_or_dict=model_cfg,
        num_geo_types=trn.get_n_geo_types(),
        max_val=max_val,
        use_cuda=trn_cfg["optimization"]["cuda"],
    )

    if trn_cfg["model"]["weights"]:
        model = load_model_chkp(model, trn_cfg["model"]["weights"], trn_cfg["optimization"]["cuda"])

    loss_fn = get_loss_fn(task)

    trainer = TRAINERS[task](
        model=model,
        train_dataloader=trn_loader,
        epochs=trn_cfg["optimization"]["epochs"],
        lr=trn_cfg["optimization"]["learning_rate"],
        loss_fn=loss_fn,
        use_cuda=trn_cfg["optimization"]["cuda"],
        test_dataloader=test_loader,
        wandb_entity=trn_cfg["metadata"]["wandb_entity"],
        wandb_project=trn_cfg["metadata"]["wandb_project"],
        wandb_group=trn_cfg["metadata"].get("wandb_group", None),
        optim_chkp=trn_cfg["model"]["weights"] if trn_cfg["optimization"]["resume"] else None,
        model_name=trn_cfg["metadata"].get("run_name", None),
        log=[trn_cfg, model_cfg],
        vis=trn_cfg["metadata"]["viz"],
        max_val=max_val
    )

    print(f"TRAINING MODEL: {task}")
    # trainer.add_file_to_chkp(fpath=trn_cfg["model"]["cfg"])
    # trainer.add_file_to_chkp(fpath=dataset_manager.train.metadata_file)
    # trainer.write_json_to_chkp(
    #     dta=,
    #     fname="normalizer.json"
    # )
    trainer.fit()


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--trn-cfg', type=str, default="./cfg_frustum_neonet.yaml", help='Path to train config file')
    parser.add_argument('--model-cfg', type=str, default=None, help='optional model cfg path override')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
