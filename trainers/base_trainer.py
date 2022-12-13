import os
import torch
from tqdm import tqdm
import logging
from wandb_utils.wandb import WandB
import numpy as np
import shutil
import json

import torch.optim as optim
import matplotlib.pyplot as plt


class RunningAvgQueue(object):
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.data=[]

    def __str__(self):
        return str(self.data)

    def add(self, x):
        self.data.append(x)
        if len(self.data) > self.maxsize:
            self.data.pop(0)

    def mean(self):
        return np.mean(self.data)


class BaseTrainer(object):

    def __init__(self, model, train_dataloader, epochs, lr, optim_type='adam',
                 test_dataloader=None, use_cuda=False, checkpoint=True, model_name=None, max_iter=None,
                 plot_iter=True, chkp_dir="./chkp", debug=False, vis=True, ma_lookback=20, weight_decay=0.0,
                 wandb_project=None, wandb_entity=None, wandb_group=None, optim_chkp=None):
        assert optim_type in ['adam']
        self.model = model
        self.cuda = use_cuda and torch.cuda.is_available()
        self.epochs = epochs
        self.lr = lr
        self.optim_type = optim_type
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.checkpoint = checkpoint
        self.plot_iter = plot_iter
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.n_gpu = torch.cuda.device_count()
        self.run_test_loop = True if self.test_dataloader is not None else False
        self.train_loss = []
        self.train_iter_loss = []
        self.train_running_avg = []
        self.test_loss = []
        self.train_eval = []
        self.test_eval = []
        self.best_loss = {"val": 1e6, "epoch": -1}
        self.max_iter = max_iter
        self.debug=debug
        self.vis=vis
        self.ma_lookback = ma_lookback
        wandb_enabled = wandb_project and wandb_entity
        self._wandb = WandB(
            project=wandb_project,
            enabled=wandb_enabled,
            entity=wandb_entity,
            name=model_name,
            job_type="TRAIN",
            group=wandb_group
        )
        self.model_name = self._wandb.run.name
        self.wandb_project=wandb_project
        self.wandb_entity=wandb_entity

        if self.cuda and self.n_gpu:
            self.model = torch.nn.DataParallel(model)
            print("Parallel processing enabled")

        self.model.to(self.device)
        self.optimizer, self.init_epoch, self.final_epoch = self.get_optimizer(
            optim_type,
            self.model.parameters(),
            lr,
            weight_decay,
            optim_chkp,
            self.cuda,
            self.epochs
        )
        self.chkp_dir = self._get_chkp_dir(self.model_name, optim_chkp, chkp_dir)

    def _get_chkp_dir(self, model_name, optim_chkp, chkp_dir):

        if not os.path.exists(chkp_dir):
            os.mkdir(chkp_dir)

        resume = True if optim_chkp else False
        if resume:
            path_splits = os.path.split(optim_chkp)
            #chkp_dir = split[0]
            model_name = path_splits[0].split("/")[-1]
            self._wandb.update_name(model_name)

        chkp_dir = self._wandb_config(resume=resume, chkp_dir=chkp_dir)

        self._init_chkp(chkp_dir)

        return chkp_dir

    @staticmethod
    def get_optimizer(optim_type, params, lr, weight_decay, optim_chkp, cuda, epochs):
        if optim_type == 'adam':
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optim_type == "sgd":
            optimizer = optim.SGD(params, lr=lr, nesterov=True, weight_decay=weight_decay)
        else:
            raise NotImplementedError(f"Optimizer type: {optim_type} not currently supported. Must be [adam, sgd]")

        init_epoch = 0
        final_epoch = epochs
        if optim_chkp:
            device = torch.device('cuda' if cuda else 'cpu')
            chkp = torch.load(optim_chkp, map_location=device)
            optimizer.load_state_dict(chkp["optimizer_state_dict"])
            init_epoch += chkp["epoch"]
            final_epoch += chkp["epoch"]
            print("Resuming training from {} (epoch: {}, loss: {:.5f}) ".format(
                optim_chkp,
                chkp["epoch"],
                chkp['loss'].cpu().data.numpy()
            )
        )

        return optimizer, init_epoch, final_epoch

    def _wandb_config(self, chkp_dir, resume=False):

        config = {
            "learning_rate": self.lr,
            "epochs": self.epochs,
            "batch_size": self.train_dataloader.batch_size,
            "optimizer": self.optim_type,
            "resume": resume
        }
        self._wandb.update_config(config)
        chkp_dir = self._wandb.get_log_dir(chkp_dir)

        return chkp_dir

    @staticmethod
    def _init_chkp(chkp_dir):
        print(f"Logging model files to {chkp_dir}")
        if not os.path.exists(chkp_dir):
            os.makedirs(chkp_dir)

    def _trn_loss(self, y_hat, targets):
        # implemented by derived class
        pass

    def _test_loss(self, y_hat, targets):
        # implemented by derived class
        pass

    def _batch_prediction(self, batch):
        # implemented by derived class
        pass

    def _trn_eval_metric(self, y_hat, y_true, **kwargs):
        # implemented by derived class
        pass

    def _test_eval_metric(self, y_hat, y_true, **kwargs):
        # implemented by derived class
        pass


    def _start_epoch(self):

        pass



    def write_model(self, model, epoch, optimizer, fname, loss):

        with open(fname, 'wb') as f:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, fname)

    def write_sub_model(self, model, fname):
        with open(fname, 'wb') as f:
            torch.save(model, f)

    def _is_best(self, loss, epoch):
        if loss < self.best_loss["val"]:
            self.best_loss["val"] = loss
            self.best_loss["epoch"] = epoch
            return True, loss
        else:
            return False, loss

    def _plot_learning_curve_iter(self, fname, running_avg, title=''):
        if self.train_iter_loss:
            x_train = range(len(self.train_iter_loss))
            plt.plot(x_train, self.train_iter_loss, label='train')
            if running_avg:
                plt.plot(range(len(running_avg)), running_avg, label="running average")
            plt.title(title)
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.legend(loc='best')
            plt.savefig(fname)
            plt.clf()
            plt.close()

    def _plot_learning_curve(self, fname, title=''):
        if self.train_loss:
            x_train = range(len(self.train_loss))
            plt.plot(x_train, self.train_loss, label='train')
            if self.run_test_loop:
                x_test = range(len(self.test_loss))
                plt.plot(x_test, self.test_loss, label="val")
            plt.title(title)
            plt.xlabel("epoch")
            plt.ylabel("error")
            plt.legend(loc='best')
            plt.savefig(fname)
            plt.clf()
            plt.close()

    def _target_to_device(self, targets):

        if isinstance(targets, list):
            for i, t in enumerate(targets):
                targets[i] = t.to(self.device)
        else:
            targets = targets.to(self.device)

        return targets

    def _checkpoint_model(self, model, epoch, epoch_loss, test_loss):
        # last checkpoint
        last_chkp_path = os.path.join(self.chkp_dir, "last.pt")
        self.write_model(self.model, epoch, self.optimizer, last_chkp_path, epoch_loss)

        # best checkpoint
        if test_loss:
            is_best, best_loss = self._is_best(test_loss.cpu().data.numpy(), epoch)
        else:
            is_best, best_loss = self._is_best(epoch_loss.cpu().data.numpy(), epoch)

        if is_best:
            best_chkp_path = os.path.join(self.chkp_dir, "best.pt")
            self.write_model(model, epoch, self.optimizer, best_chkp_path, best_loss)

    def _is_valid_loss(self, loss, y_hat, targets):
        if torch.isnan(loss):
            y_hat = y_hat.cpu()
            t = []
            for target in targets:
                t.append(target.cpu())
            msg = f"loss NaN. \n y_hat: {y_hat} \n targets: {t}"
            logging.warning(msg)

        return loss

    def _stringify_eval_dict(self, eval_dict, is_test=False):
        s = ""
        trn_tst_flag = 'val' if is_test else 'train'
        for name, eval in eval_dict.items():
            sub_str = "{}/{} {:.4f} ".format(trn_tst_flag, name, eval)
            s += sub_str
        return s

    def _increment_eval(self, epoch_dict, iter_dict):
        for k, v in iter_dict.items():
            if k not in epoch_dict:
                epoch_dict[k] = 0
            epoch_dict[k] += v
        return epoch_dict

    def _increment_eval_queue(self, epoch_dict, iter_dict):
        for k, v in iter_dict.items():
            if k not in epoch_dict:
                epoch_dict[k] = 0
            epoch_dict[k] += v
        return epoch_dict


    def add_file_to_chkp(self, fpath):

        dest_fpath = os.path.join(self.chkp_dir, fpath.split("/")[-1])
        shutil.copy(src=fpath, dst=dest_fpath)

    def write_json_to_chkp(self, dta: dict, fname: str):

        dest_fpath = os.path.join(self.chkp_dir, fname)
        with open(dest_fpath, "w") as f:
            json.dump(dta, f)

    def _get_wandb_plot(self, batch, y):
        # implemented by derived class
        pass


    def fit(self):

        if self.cuda:
            print("Training on GPU: {} devices".format(self.n_gpu))
        else:
            print("Training on CPU")

        total_iter = 0
        queue_loss = RunningAvgQueue(self.ma_lookback)


        for e in range(self.init_epoch, self.final_epoch):

            print("\n************* EPOCH: {} *************\n".format(e + 1))
            self._start_epoch()
            self.model.train()

            epoch_loss_cum = 0
            epoch_eval_cum = {}

            pbar = tqdm(self.train_dataloader, total=len(self.train_dataloader))
            i = 0
            for inputs, targets in pbar:

                if (e == 0 and i == 0) and self.vis:
                    # visualize first batch
                    self._get_wandb_plot(
                        batch=inputs,
                        y=targets.cpu()
                    )

                if self.max_iter and total_iter >= self.max_iter:
                    break

                targets = self._target_to_device(targets)

                self.optimizer.zero_grad()

                y_hat = self._batch_prediction(inputs).to(self.device)

                loss = self._trn_loss(y_hat, targets)

                if self.n_gpu > 1:
                    loss = loss.mean()

                loss.backward()
                self.optimizer.step()

                eval_dict = self._trn_eval_metric(
                    y_hat,
                    targets
                )

                epoch_loss_cum += loss.detach()
                epoch_eval_cum = self._increment_eval(epoch_eval_cum, eval_dict)

                queue_loss.add(loss.cpu().data.numpy())


                self.train_iter_loss.append(float(loss.detach().cpu().data.numpy()))
                self.train_running_avg.append(queue_loss.mean())

                pbar.set_description(
                    "Loss: {:.5f}".format(queue_loss.mean())
                )
                self._wandb.log({"train/batch-loss": queue_loss.mean()})

                i += 1
                total_iter += 1

            epoch_loss = epoch_loss_cum / len(self.train_dataloader)
            epoch_eval = {}
            for k, v in epoch_eval_cum.items():
                epoch_eval[k] = epoch_eval_cum[k] / len(self.train_dataloader)

            wb_log = {"train/loss": epoch_loss}
            for k, v in epoch_eval_cum.items():
                wb_log["train/" + k] = epoch_eval[k]

            self._wandb.log(wb_log)
            print(
                "\ntrain/loss {:.5f} ".format(epoch_loss) + self._stringify_eval_dict(epoch_eval)
            )

            self.train_loss.append(epoch_loss.cpu())

            if self.max_iter and total_iter >= self.max_iter:
                break

            # Test inference
            if self.run_test_loop:
                self.model.eval()
                print("\n----> Validation inference <----")
                pbar_test = tqdm(self.test_dataloader, total=len(self.test_dataloader))

                test_targets = []
                test_predictions = []

                for inputs, targets in pbar_test:
                    targets_batch = self._target_to_device(targets).flatten()
                    y_hat_batch = self._batch_prediction(inputs).flatten()

                    if self.n_gpu > 1:
                        test_targets.append(targets_batch.detach())
                        test_predictions.append(y_hat_batch.detach())
                    else:

                        test_targets.append(targets_batch.detach().cpu())
                        test_predictions.append(y_hat_batch.detach().cpu())

                test_targets = torch.cat(test_targets, dim=0)
                test_predictions = torch.cat(test_predictions, dim=0)

                test_loss = self._test_loss(test_predictions, test_targets)
                test_eval_dict = self._test_eval_metric(
                    test_predictions,
                    test_targets,
                    is_test=True
                )


                print("val/loss {:.5f} ".format(test_loss) + self._stringify_eval_dict(test_eval_dict, True))

                self.test_loss.append(test_loss.cpu())
                wb_log = {"val/loss": test_loss}
                for k, v in test_eval_dict.items():
                    wb_log["val/" + k] = v
                self._wandb.log(wb_log)

            if self.checkpoint:
                test_loss = test_loss if self.run_test_loop else None
                self._checkpoint_model(self.model, e, epoch_loss, test_loss)

            # plot learning curve
            if self.model_name:
                epoch_fname = f"{self.chkp_dir}/{self.model_name}_learning-curve-epochs.png"
                iter_fname = f"{self.chkp_dir}/{self.model_name}_learning-curve-iter.png"
            else:
                epoch_fname = f"{self.chkp_dir}/learning-curve-epochs.png"
                iter_fname = f"{self.chkp_dir}/learning-curve-iter.png"

            self._plot_learning_curve(epoch_fname,
                                      title='Learning Curves')
            if self.plot_iter:
                self._plot_learning_curve_iter(iter_fname,
                                               running_avg=self.train_running_avg,
                                               title='Learning Curve')

        # log model artifacts to wandb
        self._wandb.log_dir(self.chkp_dir, self.model_name, type='neonet')
