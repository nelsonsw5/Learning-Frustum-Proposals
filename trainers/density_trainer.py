import torch
import torch.nn.functional as F
from trainers.base_trainer import BaseTrainer

from eval.metrics import get_mse, get_mape, get_mae, get_acc
# from models.classifier import ClassificationDecoder
# from models.poisson import PoissonDecoder

class DensityTrainer(BaseTrainer):

    def __init__(
            self,
            model,
            train_dataloader,
            epochs,
            lr,
            loss_fn,
            optim_type='adam',
            test_dataloader=None,
            use_cuda=True,
            debug=False,
            vis=False,
            log=None,
            weight_decay=0.0,
            wandb_entity=None,
            wandb_project=None,
            wandb_group=None,
            optim_chkp=None,
            model_name=None
    ):
        self.log = log
        self.loss_fn = loss_fn


        super(DensityTrainer, self).__init__(
            model=model,
            train_dataloader=train_dataloader,
            epochs=epochs,
            lr=lr,
            optim_type=optim_type,
            test_dataloader=test_dataloader,
            use_cuda=use_cuda,
            debug=debug,
            vis=vis,
            weight_decay=weight_decay,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_group=wandb_group,
            optim_chkp=optim_chkp,
            model_name=model_name
        )


    def _batch_prediction(self, batch):

        y_hat = self.model(
            batch["points"].to(self.device),
            batch["object_geo_type"].to(self.device)
        )
        return y_hat

    def _wandb_config(self, chkp_dir, resume=False):

        config = {
            "dataset_id": self.train_dataloader.dataset.dataset_id,
            "resume": resume
        }

        if self.log:
            if isinstance(self.log, dict):
                logs = [self.log]
            else:
                logs = self.log
            for l in logs:
                for k, v in l.items():
                    config[k] = v


        self._wandb.update_config(config)
        chkp_dir = self._wandb.get_log_dir(chkp_dir)

        return chkp_dir

    def _trn_loss(self, y_hat, targets):
        return self.loss_fn(y_hat.squeeze(-1), targets.squeeze(-1))

    def _test_loss(self, y_hat, targets):
        return self.loss_fn(y_hat, targets)

    def _trn_eval_metric(self, y_hat, y_true, **kwargs):

        mape = get_mape(y_true=y_true, y_hat=y_hat)
        mae = get_mae(y_true=y_true, y_hat=y_hat)
        mse = get_mse(y_true, y_hat)

        return {'mae': mae, "mape": mape, "mse": mse}


class ClassificationTrainer(DensityTrainer):

    def __init__(
            self,
            model,
            train_dataloader,
            epochs,
            lr,
            loss_fn,
            optim_type='adam',
            test_dataloader=None,
            use_cuda=True,
            debug=False,
            vis=False,
            log=None,
            weight_decay=0.0,
            wandb_entity=None,
            wandb_project=None,
            wandb_group=None,
            optim_chkp=None,
            model_name=None,
            max_val=None
    ):
        super(ClassificationTrainer, self).__init__(
            model=model,
            train_dataloader=train_dataloader,
            epochs=epochs,
            lr=lr,
            optim_type=optim_type,
            test_dataloader=test_dataloader,
            use_cuda=use_cuda,
            debug=debug,
            vis=vis,
            weight_decay=weight_decay,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_group=wandb_group,
            optim_chkp=optim_chkp,
            model_name=model_name,
            loss_fn=loss_fn,
            log=log
        )
        self.max_val = max_val
        self.decoder = ClassificationDecoder()

    def _test_eval_metric(self, y_hat, y_true, **kwargs):
        y_true += 1  # undo decrement from dataloader

        mape = get_mape(y_true=y_true, y_hat=y_hat)
        mae = get_mae(y_true=y_true, y_hat=y_hat)
        mse = get_mse(y_true, y_hat)
        acc = get_acc(y_true=y_true, y_hat=y_hat)

        return {'mae': mae, "mape": mape, "mse": mse, "acc": acc}


    def _trn_eval_metric(self, y_hat, y_true, **kwargs):

        y_hat = self.decoder(y_hat)
        y_true += 1 # undo decrement from dataloader


        mape = get_mape(y_true=y_true, y_hat=y_hat)
        mae = get_mae(y_true=y_true, y_hat=y_hat)
        mse = get_mse(y_true, y_hat)
        acc = get_acc(y_true=y_true, y_hat=y_hat)

        return {'mae': mae, "mape": mape, "mse": mse, "acc": acc}

    def _test_loss(self, y_hat, targets):
        """
        Get the test loss for a classifier. The ClassificationDecoder class computes the
            argmax, and increments the counts by 1. Here we undo those to compute the loss.

        NOTE: This is a lower resolution estimate of the loss than the during training because
        the class probabilities are thresholded through the argmax (in ClassificationDecoder)
        @param y_hat:
        @param targets:
        @return:
        """

        # reshape if zero-padded
        if targets.ndim > 2:
            targets = targets.view(-1, 1)

        if y_hat.ndim > 2:
            y_hat = y_hat.view(-1, y_hat.shape[-1])


        y_hat = y_hat - 1 # get zero-index counts
        y_hat = F.one_hot(y_hat, num_classes=self.max_val).float()
        return self.loss_fn(y_hat, targets)

    def _trn_loss(self, y_hat, targets):

        # reshape if zero-padded
        if targets.ndim >= 2:
            targets = targets.view(-1, 1)

        if y_hat.ndim >= 2:
            y_hat = y_hat.view(-1, y_hat.shape[-1])

        return self.loss_fn(y_hat.squeeze(-1), targets.squeeze(-1))


    def _get_wandb_plot(self, batch, y):
        y_viz = y + 1  # undo decrement from dataloader

        self.train_dataloader.dataset.get_wandb_plot(
            wandb_run=self._wandb,
            batch=batch,
            y=y_viz
        )

class RegressionTrainer(DensityTrainer):

    def __init__(
            self,
            model,
            train_dataloader,
            epochs,
            lr,
            loss_fn,
            optim_type='adam',
            test_dataloader=None,
            use_cuda=True,
            debug=False,
            vis=False,
            log=None,
            weight_decay=0.0,
            wandb_entity=None,
            wandb_project=None,
            wandb_group=None,
            optim_chkp=None,
            model_name=None,
            max_val=None
    ):
        super(RegressionTrainer, self).__init__(
            model=model,
            train_dataloader=train_dataloader,
            epochs=epochs,
            lr=lr,
            optim_type=optim_type,
            test_dataloader=test_dataloader,
            use_cuda=use_cuda,
            debug=debug,
            vis=vis,
            weight_decay=weight_decay,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_group=wandb_group,
            optim_chkp=optim_chkp,
            model_name=model_name,
            loss_fn=loss_fn,
            log=log
        )

        self.max_val = max_val

    def _trn_eval_metric(self, y_hat, y_true, **kwargs):

        # undo max normalization
        if self.max_val:
            y_hat = y_hat * self.max_val
            y_true = y_true * self.max_val

        # TODO: Add flag to remove zero-pad
        keep_idx = torch.where(y_true!=0.0)
        y_true = y_true[keep_idx]
        y_hat = y_hat[keep_idx]


        mape = get_mape(y_true=y_true, y_hat=y_hat)
        mae = get_mae(y_true=y_true, y_hat=y_hat)
        mse = get_mse(y_true, y_hat)

        return {'mae': mae, "mape": mape, "mse": mse}

    def _test_eval_metric(self, y_hat, y_true, **kwargs):

        # TODO: Add flag to remove zero-pad
        keep_idx = torch.where(y_true != 0.0)
        y_true = y_true[keep_idx]
        y_hat = y_hat[keep_idx]

        mape = get_mape(y_true=y_true, y_hat=y_hat)
        mae = get_mae(y_true=y_true, y_hat=y_hat)
        mse = get_mse(y_true, y_hat)

        return {'mae': mae, "mape": mape, "mse": mse}

    def _test_loss(self, y_hat, targets):

        if self.max_val:
            # rescale to make train/val loss comparable
            y_hat = y_hat / self.max_val
            targets = targets / self.max_val

        return self.loss_fn(y_hat, targets)


    def _get_wandb_plot(self, batch, y):
        if self.max_val:
            y_viz = y * self.max_val

        self.train_dataloader.dataset.get_wandb_plot(
            wandb_run=self._wandb,
            batch=batch,
            y=y_viz
        )


class PoissonTrainer(DensityTrainer):

    def __init__(
            self,
            model,
            train_dataloader,
            epochs,
            lr,
            loss_fn,
            optim_type='adam',
            test_dataloader=None,
            use_cuda=True,
            debug=False,
            vis=False,
            log=None,
            weight_decay=0.0,
            wandb_entity=None,
            wandb_project=None,
            wandb_group=None,
            optim_chkp=None,
            model_name=None,
            max_val=None
    ):
        super(PoissonTrainer, self).__init__(
            model=model,
            train_dataloader=train_dataloader,
            epochs=epochs,
            lr=lr,
            optim_type=optim_type,
            test_dataloader=test_dataloader,
            use_cuda=use_cuda,
            debug=debug,
            vis=vis,
            weight_decay=weight_decay,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_group=wandb_group,
            optim_chkp=optim_chkp,
            model_name=model_name,
            loss_fn=loss_fn,
            log=log
        )
        self.max_val = max_val
        self.decoder = PoissonDecoder(max_val)

    def _test_eval_metric(self, y_hat, y_true, **kwargs):

        # Add flag to remove zero-pad
        keep_idx = torch.where(y_true!=0.0)
        y_true = y_true[keep_idx]
        y_hat = y_hat[keep_idx]

        mape = get_mape(y_true=y_true, y_hat=y_hat)
        mae = get_mae(y_true=y_true, y_hat=y_hat)
        mse = get_mse(y_true, y_hat)
        acc = get_acc(y_true=y_true, y_hat=y_hat)

        return {'mae': mae, "mape": mape, "mse": mse, "acc": acc}


    def _trn_eval_metric(self, y_hat, y_true, **kwargs):

        y_hat = self.decoder(y_hat)

        # TODO: Add flag to remove zero-pad
        keep_idx = torch.where(y_true!=0.0)
        y_true = y_true[keep_idx]
        y_hat = y_hat[keep_idx]

        mape = get_mape(y_true=y_true, y_hat=y_hat)
        mae = get_mae(y_true=y_true, y_hat=y_hat)
        mse = get_mse(y_true, y_hat)
        acc = get_acc(y_true=y_true, y_hat=y_hat)

        return {'mae': mae, "mape": mape, "mse": mse, "acc": acc}




TRAINERS = {
    "regressor": RegressionTrainer,
    "classifier": ClassificationTrainer,
    "poisson": PoissonTrainer,
    "other": DensityTrainer
}
