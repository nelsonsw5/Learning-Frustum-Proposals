import yaml
import os

# from models.pointpillars import PointPillars, get_n_voxels
from models.pointnet import PointNetFeatureExtractor, PointNet4DFeatureExtractor
# from models.pointnet2 import PointNet2Encoder, PointNet2Encoder4D
# from models.transformer import PointNetTransformer, Transformer

from models.model_utils import EncoderTypes


class ModelFactory(object):

    model_paths = {
        "pointnet2": os.path.join(os.path.dirname(os.path.realpath(__file__)), "cfgs/pointnet2.yaml"),
        "pointnet": os.path.join(os.path.dirname(os.path.realpath(__file__)), "cfgs/pointnet.yaml"),
        "pointpillars":os.path.join(os.path.dirname(os.path.realpath(__file__)), "cfgs/pointpillars.yaml.yaml")
    }

    def _get_yaml(self, path):
        with open(path, "r") as f:
            model_cfg = yaml.load(f, Loader=yaml.FullLoader)
        return model_cfg

    def _build_pointnet2(self, cfg, num_input_features, dropout=.0, get_4D=False):
        if get_4D:
            model = PointNet2Encoder4D(
                feature_size=num_input_features,
                sa1_dict=cfg["set_abstractor_1"],
                sa2_dict=cfg["set_abstractor_2"],
                sa3_dict=cfg["set_abstractor_3"],
                fc_layers=cfg["fc_layers"],
                dropout_p=dropout
            )
        else:

            model = PointNet2Encoder(
                feature_size=num_input_features,
                sa1_dict=cfg["set_abstractor_1"],
                sa2_dict=cfg["set_abstractor_2"],
                sa3_dict=cfg["set_abstractor_3"],
                fc_layers=cfg["fc_layers"],
                dropout_p=dropout
            )

        return model

    def _build_pointpillars(self, cfg, num_input_features, dropout=0.0):

        n_voxels = get_n_voxels(
            voxel_size=cfg["voxel_size"],
            pc_range=cfg["pc_range"]
        )
        model = PointPillars(
            bev_backone_cfg=cfg["bev_backbone"],
            num_input_channels=num_input_features,
            n_x_voxels=n_voxels[0],
            n_y_voxels=n_voxels[1],
            use_norm=cfg["batch_norm"],
            num_filters=cfg["num_filters"],
            with_distance=cfg["with_distance"],
            max_points=cfg["max_points"],
            pc_range=cfg["pc_range"],
            max_voxels=cfg["max_voxels"],
            output_feat_size=cfg["feature_size"],
            output_cfg=cfg["output_encoder"]
        )
        return model



    def build(self, model, num_input_features, cfg, dropout=0.0):
        """if not cfg:
            cfg = self._get_yaml(self.model_paths[model])"""
        cfg["backbone"] = model

        if model == EncoderTypes.POINTNET.value:
            model = PointNetFeatureExtractor(
                in_channels=num_input_features,
                feat_size=cfg["feature_size"],
                layer_dims=cfg["layer_dims"],
                dropout_prob=dropout
            )
        elif model == EncoderTypes.POINTNET_TRANSFORMER.value:
            pointnet = PointNet4DFeatureExtractor(
                in_channels=num_input_features,
                feat_size=cfg["feature_size"],
                layer_dims=cfg["layer_dims"],
                dropout_prob=dropout
            )
            transformer = Transformer(
                dim=cfg["feature_size"],
                attn_heads=cfg["transformer"]["attn_heads"],
                dropout_prob=dropout,
                n_enc_blocks=cfg["transformer"]["n_encoder_blocks"],
                use_global=cfg["transformer"]["use_global"]
            )
            model = PointNetTransformer(
                pointnet=pointnet,
                transformer=transformer
            )
        elif model == EncoderTypes.POINTNET2.value:
             model = self._build_pointnet2(cfg, num_input_features)

        elif model == EncoderTypes.POINTNET2_TRANSFORMER.value:
            pointnet = self._build_pointnet2(cfg, num_input_features, get_4D=True)
            transformer = Transformer(
                dim=cfg["feature_size"],
                attn_heads=cfg["transformer"]["attn_heads"],
                dropout_prob=dropout,
                n_enc_blocks=cfg["transformer"]["n_encoder_blocks"],
                use_global=cfg["transformer"]["use_global"]
            )
            model = PointNetTransformer(
                pointnet=pointnet,
                transformer=transformer
            )
        else:
            raise NotImplementedError()
        return model, cfg


