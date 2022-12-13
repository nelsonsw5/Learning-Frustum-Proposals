import numpy as np
from dataset_manager.dataset_manager import DatasetManager

def setup(trn_cfg, model_cfg):

    dataset_manager = DatasetManager(trn_cfg["dataset"]["dir"])

    print("Total items:", len(dataset_manager))

    print("Filtering All Ones")
    dataset_manager.filter_all_ones()

    print("Filtering Scenes with 0 counts")
    dataset_manager.filter_scenes_with_zero_counts()

    print("Filtering point clouds with missing data")
    dataset_manager.filter_missing_data()

    print("Total items after filtering:", len(dataset_manager))

    trn_cfg["dataset"].update(
        {
            'n_examples': {
                "train": len(dataset_manager.train),
                "val": len(dataset_manager.val)
            },
            'hash': {
                "train": hash(dataset_manager.train),
                "val": hash(dataset_manager.val),
                "test": hash(dataset_manager.test)

            }
        }
    )

    # By default cacheing is set to false. Uncomment this line to set change the cacheing policy
    # dataset_manager.set_cache_policy(cache_labels=True, cache_centroids=True, cache_point_clouds=False)

    return dataset_manager, trn_cfg, model_cfg


def get_n_params(model):
    n_params = 0
    for layer in model.parameters():
        dims = layer.size()
        cnt = dims[0]
        for d in dims[1:]:
            cnt *= d
        n_params += cnt

    return n_params


def get_fc_layer_depth(layer_dims):
    return len(layer_dims)

def get_max_dim(layer_dims):
    return np.max(np.array(layer_dims))

def get_model_stats(model, cfg):
    log_dict = {
        "n_params": get_n_params(model),
        "fc_layer_depth": get_fc_layer_depth(
            cfg["head"]["fc_layers"]
        ),
        "fc_layer_max_dim": get_max_dim(
            cfg["head"]["fc_layers"]
        ),
        "point_encoder_depth": get_fc_layer_depth(
            cfg["point_encoder"]["layer_dims"]
        ),
        "point_encoder_max_dim": get_max_dim(
            cfg["point_encoder"]["layer_dims"]
        )
    }

    return log_dict