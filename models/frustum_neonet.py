import torch
from torch import nn
import yaml
import pdb

from models.model_factory import ModelFactory
from models.model_utils import NeonetTypes
from models.regressor import RegressionDecoder, RegressorHead

from utils import print_yaml
from train.train_utils import get_model_stats

import typing

class FrustumNeonet(nn.Module):

    def __init__(self, pc_encoder, output_head, decoder, use_cuda):
        super(FrustumNeonet, self).__init__()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.encoder = pc_encoder
        self.output_head = output_head
        self.decoder = decoder

        if use_cuda:
            self = self.cuda()


    def forward(self, points, geo_type):
        h = self.encoder(points)
        h = torch.cat([h, geo_type], dim=-1)
        y_hat = self.output_head(h)

        if not self.training:
            y_hat = self.decoder(y_hat)

        return y_hat





def get_output_head(model_cfg, num_geo_types, max_val):

    n_reg_feats = model_cfg["point_encoder"]["feature_size"] + num_geo_types
    if model_cfg["head"]["type"] == NeonetTypes.REGRESSOR.value:
        head = RegressorHead(
            input_size=n_reg_feats,
            reg_layers=model_cfg["head"]["fc_layers"],
            n_classes=1,
            dropout_prob=model_cfg["meta"]["dropout"]
        )

        decoder = RegressionDecoder(max_val)

    else:
        raise NotImplementedError(model_cfg["head"]["type"])

    return head, decoder

def build_from_yaml(fpath_or_dict: typing.Union[str, dict], num_geo_types, use_cuda, max_val):

    """

    @param fpath_or_dict: (str) filepath to model cfg, or dictionary of config
    @param num_geo_types: (int) number of geometric types
    @param use_cuda: (bool) training with cuda
    @param max_val: (int) maximum target value observed in dataset
    @return:
    """
    if isinstance(fpath_or_dict, str):
        with open(fpath_or_dict, "r") as f:
            neonet_cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        neonet_cfg = fpath_or_dict

    print("MODEL SETTINGS:")
    print_yaml(neonet_cfg)

    num_channels = neonet_cfg["point_encoder"]["num_input_channels"]

    model_builder = ModelFactory()
    pc_encoder, _ = model_builder.build(
        model=neonet_cfg["point_encoder"]["type"],
        num_input_features=num_channels,
        cfg=neonet_cfg["point_encoder"],
        dropout=neonet_cfg["meta"]["dropout"]
    )

    head, decoder = get_output_head(neonet_cfg, num_geo_types, max_val)

    model = FrustumNeonet(
        pc_encoder=pc_encoder,
        output_head=head,
        decoder=decoder,
        use_cuda=use_cuda
    )
    log_stats = get_model_stats(model, neonet_cfg)
    neonet_cfg.update(log_stats)

    print(f"# of Model Parameters: {neonet_cfg['n_params']}")
    return model, neonet_cfg


def get_loss_fn(neonet_type):
    if neonet_type == NeonetTypes.REGRESSOR.value:
        loss_fn = nn.MSELoss()
    else:
        print("unrecognized neonet type")
    return loss_fn