import os
from tqdm import tqdm
from typing import Dict
import numpy as np
import torch

from torch.utils.data import DataLoader
import torch.nn.functional as F

from eval.metrics import get_mse, get_mape, get_mae, get_acc, get_acc_k
from evaluators.evaluators import Evaluator

from models.classifier import ClassificationDecoder
from models.poisson import PoissonDecoder
from models.regressor import RegressionDecoder

from preprocessing.preprocess_utils import get_iou, get_lab_set

class DensityEvaluator(Evaluator):

    def __init__(
            self,
            model,
            test_dataloader,
            cuda,
            gen_plots,
            output_dir,
            decoder,
            loss_fn=None,
            log_wandb=False,
            wandb_prj=None,
            wandb_entity=None,
            wandb_group=None,
            wandb_run_name=None,
            limit=None,
            log=None,
            max_val=None,
    ):
        super(DensityEvaluator, self).__init__(
            model=model,
            test_dataloader=test_dataloader,
            cuda=cuda,
            gen_plots=gen_plots,
            output_dir=output_dir,
            wandb_enabled=log_wandb,
            wandb_prj=wandb_prj,
            wandb_entity=wandb_entity,
            wandb_group=wandb_group,
            limit=limit,
            log=log,
            run_name=wandb_run_name
        )
        self.decoder = decoder
        self.max_val = max_val
        self.loss_fn = loss_fn

    def _batch_plot(self, y_hat, y_true, eval_dict, loss, cls_labels, **kwargs):
        pass

    def _batch_prediction(self, batch):

        with torch.no_grad():
            y_hat = self.model(
                batch["points"].to(self.device),
                batch["object_geo_type"].to(self.device)
            )
        return y_hat

    def _loss(self, y_hat, targets):
        if self.loss_fn:
            return self.loss_fn(y_hat, targets)
        else:
            return None


    def _eval_metric(self, y_hat, y_true, **kwargs):

        # accuracy
        acc = get_acc(y_true=y_true, y_hat=y_hat)
        acc_k = get_acc_k(y_true, y_hat, k=1)
        # mean
        mape = get_mape(
            y_true=y_true,
            y_hat=y_hat,
            zero_correct=kwargs.get("zero_correct", False)
        )
        mae = get_mae(y_true=y_true, y_hat=y_hat)
        mse = get_mse(y_true, y_hat)


        metrics = {
            f"{kwargs['prefix']}/mae": mae,
            f"{kwargs['prefix']}/mape": mape,
            f"{kwargs['prefix']}/mse": mse,
            f"{kwargs['prefix']}/acc": acc,
            f"{kwargs['prefix']}/acc@3": acc_k
        }

        if kwargs.get("median", False):
            # median
            med_ape = get_mape(
                y_true=y_true,
                y_hat=y_hat,
                zero_correct=kwargs.get("zero_correct", False),
                median=kwargs.get("median", False)
            )
            med_ae = get_mae(y_true=y_true, y_hat=y_hat, median=kwargs.get("median", False))
            med_se = get_mse(y_true, y_hat, median=kwargs.get("median", False))

            metrics.update(
                {
                    f"{kwargs['prefix']}/med-abs-err": med_ae,
                    f"{kwargs['prefix']}/med-abs-pct-err": med_ape,
                    f"{kwargs['prefix']}/med-square-err": med_se,
                }
            )

        return metrics

    def _scene_eval_metrics(self, y_hat, y_true, **kwargs):
        n_centroids = y_true.shape[0]

        y_hat = self.decoder(y_hat)
        y_true = self.decoder(y_true)

        resid = y_true - y_hat
        cum_pct_error = torch.abs(resid).sum() / y_true.sum()

        cls_idx = (torch.argmax(kwargs["cls"].cpu(), dim=-1) + 1).unsqueeze(-1)

        cls_dict = {}
        for i in range(n_centroids):
            cls = cls_idx.data.numpy()[i][0]
            if cls not in cls_dict:
                cls_dict[cls] = {'y_true': 0, 'y_hat': 0}

            y_hat_i = y_hat.data.numpy()[i][0]
            y_true_i = y_true.data.numpy()[i][0]

            cls_dict[cls]['y_true'] += y_true_i
            cls_dict[cls]['y_hat'] += y_hat_i

        abs_error = []
        abs_pct_error = []
        for k, v in cls_dict.items():
            res = v["y_true"] - v["y_hat"]
            abs_error.append(
                abs(res)
            )
            abs_pct_error.append(
                abs(res / v["y_true"])
            )

        return cum_pct_error.data.numpy(), np.array(abs_error), np.array(abs_pct_error)

    def _set_dataloader(self, loader):
        self.test_dataloader = loader

    def get_before_after_count(self, before: dict, after: dict):
        """

        @param before:
        @param after:
        @return:
        """

        keys = set(before.keys()).union(set(before.keys()))
        output = {}
        for k in keys:
            before_count = before.get(k, 0)
            after_count = after.get(k, 0)
            output[k] = after_count - before_count

        return output

    def _count_dicts_totensor(self, targets: dict, preds: dict):
        """

        @param targets:
        @param preds:
        @return:
        """
        assert targets.keys() == preds.keys()
        n = len(targets)
        key_to_idx = dict(
            zip(
                list(targets.keys()),
                range(n)
            )
        )

        target_tensor = torch.zeros(n)
        preds_tensor = torch.zeros(n)

        for k, c_t in targets.items():
            c_p = preds[k]

            target_tensor[key_to_idx[k]] = c_t
            preds_tensor[key_to_idx[k]] = c_p

        return target_tensor, preds_tensor

    def diff_eval(self, num_cands: int = 5):

        self.model.eval()
        print("Differencing Evaluation:")

        dta = self.test_dataloader.dataset.dataset
        n = len(dta)
        iou_mtx = np.zeros((n, n))

        print("Computing distance matrix:")
        for i in tqdm(range(n)):
            scene_i = dta[i]
            labs_i = get_lab_set(scene_i.label.path)
            for j in range(n):
                scene_j = dta[j]
                labs_j = get_lab_set(scene_j.label.path)

                iou = get_iou(labs_i, labs_j)
                iou_mtx[i, j] = iou

        # get top candidates (excluding self)
        candidates = np.flip(np.argsort(iou_mtx)[:, -(num_cands+1):-1], axis=1)

        total = self.limit if self.limit else len(self.test_dataloader)
        pbar_test = tqdm(self.test_dataloader, total=total)

        target_all = []
        pred_all = []
        print("Evaluating Before/After candidates")
        i = 0
        for inputs, targets in pbar_test:

            y_hat_batch = self._batch_prediction(inputs).flatten()

            targets_batch = self.target_to_device(targets, self.device).flatten()
            targets_batch = self._rescale_target(targets_batch)

            stratified = self._get_stratified_counts(y_hat_batch, targets_batch, inputs["upcs"])

            strat_pred = {}
            strat_true = {}
            for k, v in stratified.items():
                strat_true[k] = v["targets"]
                strat_pred[k] = v["preds"]


            for j in range(num_cands):
                cand_idx = candidates[i, j]
                cand_iou = iou_mtx[i, cand_idx]

                if cand_iou == 0.0:
                    # skip candidate if no class overlap
                    continue

                x_cand, y_cand = self.test_dataloader.dataset[cand_idx]

                x_cand["points"] = x_cand["points"].unsqueeze(0)
                x_cand["object_geo_type"] = x_cand["object_geo_type"].unsqueeze(0)

                y_hat_cand = self._batch_prediction(x_cand).flatten()
                targets_cand = self.target_to_device(y_cand, self.device).flatten()
                targets_cand = self._rescale_target(targets_cand)

                cand_stratified = self._get_stratified_counts(y_hat_cand, targets_cand, x_cand["upcs"])
                cand_strat_pred = {}
                cand_strat_true = {}

                for k, v in cand_stratified.items():
                    cand_strat_true[k] = v["targets"]
                    cand_strat_pred[k] = v["preds"]


                ## get before after count
                ## we want to "more dense" display to act as the 'after' count
                if targets_batch.sum() > targets_cand.sum():
                    # candidate is before, target is after
                    gt_diff = self.get_before_after_count(
                        before=cand_strat_true,
                        after=strat_true
                    )
                    pred_diff = self.get_before_after_count(
                        before=cand_strat_pred,
                        after=strat_pred
                    )
                else:
                    # target is before, candiate is after
                    gt_diff = self.get_before_after_count(
                        before=strat_true,
                        after=cand_strat_true
                    )
                    pred_diff = self.get_before_after_count(
                        before=strat_pred,
                        after=cand_strat_pred
                    )

                gt_diff, pred_diff = self._count_dicts_totensor(
                    gt_diff,
                    pred_diff
                )


                pred_all.append(pred_diff)
                target_all.append(gt_diff)

                cand_eval = self._eval_metric(
                    pred_diff,
                    gt_diff,
                    prefix="diff-index",
                    zero_correct=True
                )

                log_dict = {
                    "diff-index/cand-iou": cand_iou,
                    "diff-index/cand-mae": cand_eval["diff-index/mae"]
                }
                self._wandb.log(log_dict)


            i += 1

        pred_all = torch.cat(pred_all)
        target_all = torch.cat(target_all)

        log_dict = self._eval_metric(
            pred_all,
            target_all,
            prefix="diff",
            zero_correct=True,
            median=True
        )
        self._wandb.log(log_dict)


class RegressionEvaluator(DensityEvaluator):

    def __init__(
            self,
            model,
            test_dataloader,
            cuda,
            gen_plots,
            output_dir,
            log_wandb=False,
            wandb_prj=None,
            wandb_entity=None,
            wandb_group=None,
            wandb_run_name=None,
            limit=None,
            log=None,
            max_val=None,
            loss_fn=None,
    ):
        super(RegressionEvaluator, self).__init__(
            model=model,
            test_dataloader=test_dataloader,
            cuda=cuda,
            gen_plots=gen_plots,
            output_dir=output_dir,
            decoder=RegressionDecoder(max_val=max_val),
            log_wandb=log_wandb,
            wandb_prj=wandb_prj,
            wandb_entity=wandb_entity,
            wandb_group=wandb_group,
            limit=limit,
            log=log,
            loss_fn=loss_fn,
            wandb_run_name=wandb_run_name
        )
        self.max_val = max_val

    def _get_wandb_plot(self, wandb_run, eval_dict, batch, y, y_hat):
        #if self.max_val:
        #    y_viz = y * self.max_val
        #else:
        #    y_viz = y

        self.test_dataloader.dataset.get_wandb_plot(
            wandb_run=wandb_run,
            batch=batch,
            y=y,
            y_hat=y_hat,
            eval_dict=eval_dict
        )

    """def _rescale_target(self, target: torch.Tensor):
        if self.max_val:
            return target * self.max_val
        else:
            return """






class ClassificationEvaluator(DensityEvaluator):

    def __init__(
            self,
            model,
            test_dataloader,
            cuda,
            gen_plots,
            output_dir,
            log_wandb=False,
            wandb_prj=None,
            wandb_entity=None,
            wandb_group=None,
            wandb_run_name=None,
            limit=None,
            log=None,
            max_val=None,
            loss_fn=None,
    ):
        super(ClassificationEvaluator, self).__init__(
            model=model,
            test_dataloader=test_dataloader,
            cuda=cuda,
            gen_plots=gen_plots,
            output_dir=output_dir,
            decoder=ClassificationDecoder(),
            log_wandb=log_wandb,
            wandb_prj=wandb_prj,
            wandb_entity=wandb_entity,
            wandb_group=wandb_group,
            wandb_run_name = wandb_run_name,
            limit=limit,
            log=log,
            loss_fn=loss_fn
        )
        self.max_val = max_val


    def _rescale_target(self, target: torch.Tensor):
        # undo scaling 0-index scaling
        return target + 1




    def _loss(self, y_hat, targets):
        """
        Get the test loss for a classifier. The ClassificationDecoder class computes the
            argmax, and increments the counts by 1. Here we undo those to compute the loss.

        NOTE: This is a lower resolution estimate of the loss than the during training because
        the class probabilities are thresholded through the argmax (in ClassificationDecoder)
        @param y_hat:
        @param targets:
        @return:
        """
        y_hat = y_hat - 1 # get zero-index counts
        y_hat = F.one_hot(y_hat, num_classes=self.max_val).float()
        return self.loss_fn(y_hat, targets)


    def _get_wandb_plot(self, wandb_run, eval_dict, batch, y, y_hat):
        y_viz = y + 1  # undo decrement from dataloader

        self.test_dataloader.dataset.get_wandb_plot(
            wandb_run=wandb_run,
            batch=batch,
            y=y_viz,
            y_hat=y_hat,
            eval_dict=eval_dict
        )

class PoissonEvaluator(DensityEvaluator):

    def __init__(
            self,
            model,
            test_dataloader,
            cuda,
            gen_plots,
            output_dir,
            log_wandb=False,
            wandb_prj=None,
            wandb_entity=None,
            wandb_group=None,
            wandb_run_name=None,
            limit=None,
            log=None,
            max_val=None,
            loss_fn=None
    ):
        super(PoissonEvaluator, self).__init__(
            model=model,
            test_dataloader=test_dataloader,
            cuda=cuda,
            gen_plots=gen_plots,
            output_dir=output_dir,
            decoder=PoissonDecoder(max_val),
            log_wandb=log_wandb,
            wandb_prj=wandb_prj,
            wandb_entity=wandb_entity,
            wandb_group=wandb_group,
            limit=limit,
            log=log,
            loss_fn=loss_fn,
            wandb_run_name=wandb_run_name
        )
        self.max_val = max_val


EVALUATORS = {
    "regressor": RegressionEvaluator,
    "classifier": ClassificationEvaluator,
    "poisson": PoissonEvaluator,
    "other": DensityEvaluator
}
