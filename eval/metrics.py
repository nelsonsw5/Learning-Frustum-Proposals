import torch

# Accuracy

def get_acc(y_true, y_hat):

    is_eq = (y_true == y_hat).float()
    acc = torch.mean(is_eq)
    return acc

def get_acc_k(y_true, y_hat, k):
    acc_k = torch.where((y_hat <= y_true + k) & (y_hat >= y_true - k), 1, 0).float().mean()
    return acc_k
# MEAN SQUARED ERROR
def get_mse(y_true, y_hat, median: bool = False):
    res = (y_true - y_hat).float()

    if median:
        mse = torch.median(res.pow(2))
    else:
        mse = torch.mean(res.pow(2))

    return mse


# MEAN ABSOLUTE PERCENTAGE ERROR
def get_mape(y_true: torch.Tensor, y_hat: torch.Tensor, zero_correct: bool = False, median: bool = False):

    res = (y_true - y_hat).float()
    res_pct = (res / y_true).abs()

    if zero_correct:
        # both y hat and y true are zero --> set to 0
        idx = torch.where((y_true == 0.0) & (y_hat == 0.0))[0]
        res_pct[idx] = 0

        # only y_true is 0, y_hat is nonzero
        idx = torch.where((y_true == 0.0) & (y_hat != 0.0))[0]
        res_pct[idx] = (((y_true[idx] + 1) - (y_hat[idx] + 1)).float() / (y_true[idx] + 1)).abs()

    if median:
        mape = torch.median(res_pct)
    else:
        mape = torch.mean(res_pct)
    return mape


def get_mae(y_true, y_hat, median: bool = False):
    res = (y_true - y_hat).float().abs()
    if median:
        mae = torch.median(res)
    else:
        mae = torch.mean(res)
    return mae


