import torch
from torch import nn
from collections import OrderedDict

from transforms import MaxDeNormalizer

class RegressorHead(nn.Module):
    """
        Regressor with Fully Connected Layers
    """

    def __init__(self, input_size, reg_layers, n_classes, dropout_prob):
        super(RegressorHead, self).__init__()
        self.reg_layers = reg_layers
        self.input_size = input_size
        #self.dropout = nn.Dropout(p=dropout_prob)
        self.dropout_prob = dropout_prob
        self.weights = self._build_mlp(input_size, reg_layers)
        self.output = nn.Linear(self.reg_layers[-1], n_classes)

    def _build_mlp(self, input_size, reg_layers):
        W = []
        for i, dim in enumerate(reg_layers):
            layer_name = f"layer_{i}"
            W.append((f"dropout_{i}", nn.Dropout(p=self.dropout_prob)))
            if i == 0:
                w = nn.Linear(input_size, dim)
            else:
                w = nn.Linear(reg_layers[i - 1], dim)
            W.append((layer_name, w))
            W.append((f"relu_{i}", nn.ReLU()))

        weights = nn.Sequential(
            OrderedDict(W)
        )
        return weights


    def forward(self, x):
        h = self.weights(x)
        y_hat = self.output(h)
        return y_hat


class RegressionDecoder(object):

    def __init__(self, max_val, round=True):
        self.denorm = MaxDeNormalizer(max=max_val)
        self.round = round

    def __call__(self, x,  *args, **kwargs):

        x = self.denorm(x)

        if self.round:
            x = torch.round(x)

        return x
