
import torch
from wandb_utils.wandb import WandB
import numpy as np



run = WandB(
        project='3DViz',
        enabled=True,
        entity='nelsonsw5',
        name='test',
        job_type="Vizualize"
        )


pc = torch.tensor([
    # back     left     bottom
    [ 1.2831,  3.9810, -1.3087],
    # back     left     top
    [ 1.2831,  3.9810,  0.2779],
    # forward  left     bottom
    [-0.7898,  3.4657, -1.3087],
    # forward  left     top
    [-0.7898,  3.4657,  0.2779],
    # back     right    bottom
    [ 1.8631,  1.6476, -1.3087],
    # back     right    top
    [ 1.8631,  1.6476,  0.2779],
    # forward  right    bottom
    [-0.2098,  1.1324, -1.3087],
    # forward  right    top
    [-0.2098,  1.1324,  0.2779]
        ])

# abs(torch.unique(np.split(pc,2)[0][:,0])[0] - torch.unique(np.split(pc,2)[0][:,0])[1])
    
pc_dict = run.get_point_cloud_log(pc)
dict_list = [pc_dict]
log_dict = {}
for d in dict_list:
    for k, v in d.items():
        log_dict[k] = v
run.log(log_dict)