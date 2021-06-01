import os

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from operators import FanbeamRadon, l2_error


# ----- ----- Abstract Base Network ----- -----


class InvNet(torch.nn.Module, metaclass=ABCMeta):
    """ Abstract base class for networks solving linear inverse problems.

    The network is intended for the denoising of a direct inversion of a 2D
    signal from (noisy) linear measurements. The measurement model

        y = Ax + noise

    can be used to obtain an approximate reconstruction x_ from y using, e.g.,
    the pseudo-inverse of A. The task of the network is either to directly
    obtain x from y or denoise and improve this first inversion x_ towards x.

    """

    def __init__(self):
        super(InvNet, self).__init__()

    @abstractmethod
    def forward(self, z):
        """
        Applies the network to a batch of inputs z, either y or x_ or both.
        """
        pass

    def freeze(self):
        """ Freeze all model weights, i.e. prohibit further updates. """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """ Unfreeze all model weights, i.e. allow further updates. """
        for param in self.parameters():
            param.requires_grad = True

    @property
    def device(self):
        return next(self.parameters()).device

    def _train_step(
        self,
        batch_idx,
        batch,
        loss_func,
        optimizer,
        scaler,
        batch_size,
        acc_steps,
    ):
        with torch.cuda.amp.autocast(enabled=self.mixed_prec):
            if len(batch) == 2:
                inp, tar = batch
                inp = inp.to(self.device)
                tar = tar.to(self.device)
                pred = self.forward(inp)
            else:
                inp, aux, tar = batch
                inp = inp.to(self.device)
                aux = aux.to(self.device)
                tar = tar.to(self.device)
                pred = self.forward((inp, aux))
            loss = loss_func(pred, tar) / acc_steps
        scaler.scale(loss).backward()
        if (batch_idx // batch_size + 1) % acc_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        return loss * acc_steps, inp, tar, pred

    def _val_step(self, batch_idx, batch, loss_func):
        if len(batch) == 2:
            inp, tar = batch
            inp = inp.to(self.device)
            tar = tar.to(self.device)
            pred = self.forward(inp)
        else:
            inp, aux, tar = batch
            inp = inp.to(self.device)
            aux = aux.to(self.device)
            tar = tar.to(self.device)
            pred = self.forward((inp, aux))
        loss = loss_func(pred, tar)
        return loss, inp, tar, pred

    def _on_epoch_end(
        self,
        epoch,
        save_epochs,
        save_path,
        logging,
        loss,
        inp,
        tar,
        pred,
        v_loss,
        v_inp,
        v_tar,
        v_pred,
        val_data,
        rel_err_val,
        chall_err_val,
    ):

        self._print_info()

        logging = logging.append(
            {
                "loss": loss.item(),
                "val_loss": v_loss.item(),
                "rel_l2_error": l2_error(
                    pred, tar, relative=True, squared=False
                )[0].item(),
                "val_rel_l2_error": rel_err_val,
                "chall_err": l2_error(
                    pred, tar, relative=False, squared=False
                )[0].item()
                / np.sqrt(pred.shape[-1] * pred.shape[-2]),
                "val_chall_err": chall_err_val,
            },
            ignore_index=True,
            sort=False,
        )

        print(logging.tail(1))

        if (epoch + 1) % save_epochs == 0:
            fig = self._create_figure(
                logging, loss, inp, tar, pred, v_loss, v_inp, v_tar, v_pred
            )

            os.makedirs(save_path, exist_ok=True)
            torch.save(
                self.state_dict(),
                os.path.join(
                    save_path, "model_weights_epoch{}.pt".format(epoch + 1)
                ),
            )
            logging.to_pickle(
                os.path.join(
                    save_path, "losses_epoch{}.pkl".format(epoch + 1)
                ),
            )
            fig.savefig(
                os.path.join(save_path, "plot_epoch{}.png".format(epoch + 1)),
                bbox_inches="tight",
            )

        return logging

    def _create_figure(
        self, logging, loss, inp, tar, pred, v_loss, v_inp, v_tar, v_pred
    ):
        def _implot(sub, im):
            if im.shape[-3] == 2:  # complex image
                p = sub.imshow(
                    torch.sqrt(im.pow(2).sum(-3))[0, :, :].detach().cpu()
                )
            else:  # real image
                p = sub.imshow(im[0, 0, :, :].detach().cpu())
            return p

        fig, subs = plt.subplots(2, 3, clear=True, num=1, figsize=(15, 10))

        # training and validation loss
        subs[0, 0].set_title("losses")
        subs[0, 0].semilogy(logging["loss"], label="train")
        subs[0, 0].semilogy(logging["val_loss"], label="val")
        subs[0, 0].legend()

        # training and validation challenge-loss
        subs[0, 1].set_title("challenge metrics")
        subs[0, 1].semilogy(logging["chall_err"], label="train")
        subs[0, 1].semilogy(logging["val_chall_err"], label="val")
        subs[0, 1].legend()

        # validation input
        p10 = _implot(subs[1, 0], v_inp)
        subs[1, 0].set_title("val input")
        plt.colorbar(p10, ax=subs[1, 0])

        # validation output
        p11 = _implot(subs[1, 1], v_pred)
        subs[1, 1].set_title(
            "val:\n ||x0-xr||_2 / ||x0||_2 = \n "
            "{:1.2e}".format(logging["val_rel_l2_error"].iloc[-1])
        )
        plt.colorbar(p11, ax=subs[1, 1])

        # validation difference
        p12 = _implot(subs[1, 2], v_pred - v_tar)
        subs[1, 2].set_title(
            "val diff: x0 - x_pred \n val_chall="
            "{:1.2e}".format(logging["val_chall_err"].iloc[-1])
        )
        plt.colorbar(p12, ax=subs[1, 2])

        # training output
        p02 = _implot(subs[0, 2], pred)
        subs[0, 2].set_title(
            "train:\n ||x0-xr||_2 / ||x0||_2 = \n "
            "{:1.2e}".format(logging["rel_l2_error"].iloc[-1])
        )
        plt.colorbar(p02, ax=subs[0, 2])

        return fig

    def _add_to_progress_bar(self, dict):
        """ Can be overwritten by child classes to add to progress bar. """
        return dict

    def _on_train_end(self, save_path, logging):
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            self.state_dict(), os.path.join(save_path, "model_weights.pt")
        )
        logging.to_pickle(os.path.join(save_path, "losses.pkl"),)

    def _print_info(self):
        """ Can be overwritten by child classes to print at epoch end. """
        pass

    def train_on(
        self,
        train_data,
        val_data,
        num_epochs,
        batch_size,
        loss_func,
        save_path,
        save_epochs=50,
        optimizer=torch.optim.Adam,
        optimizer_params={"lr": 2e-4, "eps": 1e-3},
        scheduler=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 1, "gamma": 1.0},
        acc_steps=1,
        train_transform=None,
        val_transform=None,
        train_loader_params={"shuffle": True},
        val_loader_params={"shuffle": False},
        mixed_prec=False,
    ):
        self.mixed_prec = mixed_prec
        scaler = torch.cuda.amp.GradScaler(enabled=mixed_prec)
        optimizer = optimizer(self.parameters(), **optimizer_params)
        scheduler = scheduler(optimizer, **scheduler_params)

        if isinstance(train_data, torch.utils.data.ConcatDataset):
            for ds in train_data.datasets:
                ds.transform = train_transform
        else:
            train_data.transform = train_transform
        if isinstance(val_data, torch.utils.data.ConcatDataset):
            for ds in val_data.datasets:
                ds.transform = val_transform
        else:
            val_data.transform = val_transform

        train_loader_params = dict(train_loader_params)
        val_loader_params = dict(val_loader_params)
        if "sampler" in train_loader_params:
            train_loader_params["sampler"] = train_loader_params["sampler"](
                train_data
            )
        if "sampler" in val_loader_params:
            val_loader_params["sampler"] = val_loader_params["sampler"](
                val_data
            )

        data_load_train = torch.utils.data.DataLoader(
            train_data, batch_size, **train_loader_params
        )
        data_load_val = torch.utils.data.DataLoader(
            val_data, batch_size, **val_loader_params
        )

        logging = pd.DataFrame(
            columns=["loss", "val_loss", "rel_l2_error", "val_rel_l2_error"]
        )

        for epoch in range(num_epochs):
            self.train()  # make sure we are in train mode
            t = tqdm(
                enumerate(data_load_train),
                desc="epoch {} / {}".format(epoch, num_epochs),
                total=-(-len(train_data) // batch_size),
                disable="SGE_TASK_ID" in os.environ,
            )
            optimizer.zero_grad()
            loss = 0.0
            for i, batch in t:
                loss_b, inp, tar, pred = self._train_step(
                    i,
                    batch,
                    loss_func,
                    optimizer,
                    scaler,
                    batch_size,
                    acc_steps,
                )
                t.set_postfix(
                    **self._add_to_progress_bar({"loss": loss_b.item()})
                )
                loss += loss_b
            loss /= i + 1

            with torch.no_grad():
                self.eval()  # make sure we are in eval mode
                scheduler.step()
                v_loss = 0.0
                rel_err_val = 0.0
                chall_err_val = 0.0
                for i, v_batch in enumerate(data_load_val):
                    v_loss_b, v_inp, v_tar, v_pred = self._val_step(
                        i, v_batch, loss_func
                    )
                    rel_err_val += l2_error(
                        v_pred, v_tar, relative=True, squared=False
                    )[0].item()
                    chall_err_val += l2_error(
                        v_pred, v_tar, relative=False, squared=False
                    )[0].item() / np.sqrt(v_pred.shape[-1] * v_pred.shape[-2])
                    v_loss += v_loss_b
                v_loss /= i + 1
                rel_err_val /= i + 1
                chall_err_val /= i + 1

                logging = self._on_epoch_end(
                    epoch,
                    save_epochs,
                    save_path,
                    logging,
                    loss,
                    inp,
                    tar,
                    pred,
                    v_loss,
                    v_inp,
                    v_tar,
                    v_pred,
                    val_data,
                    rel_err_val,
                    chall_err_val,
                )

        self._on_train_end(save_path, logging)
        return logging


# ----- ----- Trainable Radon Op ----- -----
class RadonNet(InvNet):
    def __init__(
        self,
        n,
        angles,
        scale,
        d_source,
        n_detect,
        s_detect,
        mode="fwd",
        **kwargs,
    ):
        super(RadonNet, self).__init__()
        self.mode = mode
        self.OpR = FanbeamRadon(
            n, angles, scale, d_source, n_detect, s_detect, **kwargs,
        )

    @classmethod
    def new_from_state_dict(cls, state, **kwargs):
        state_init = {k[4:]: v for (k, v) in state.items()}
        del state_init["m"]
        del state_init["inv_scale"]
        del state_init["fwd_offset"]
        state_init.update(kwargs)
        net = cls(**state_init)
        net.load_state_dict(state)
        return net

    def forward(self, inp):
        if self.mode == "fwd":
            out = self.OpR.dot(inp)
        elif self.mode == "bwd":
            out = self.OpR.inv(inp)
        elif self.mode == "both":
            inp1, inp2 = inp
            out = self.OpR.dot(inp1), self.OpR.inv(inp2)
        elif self.mode == "chain":
            out1 = self.OpR.dot(inp)
            out2 = self.OpR.inv(out1)
            out = (out1, out2)
        return out

    def _print_info(self):
        print("Current parameters(s):")
        print(list(self.parameters()))

    def _train_step(
        self,
        batch_idx,
        batch,
        loss_func,
        optimizer,
        scaler,
        batch_size,
        acc_steps,
    ):
        with torch.cuda.amp.autocast(enabled=False):
            if self.mode == "fwd" or self.mode == "bwd":
                inp, tar = batch
                inp = inp.to(self.device)
                tar = tar.to(self.device)
            elif self.mode == "both":
                inp1, inp2, tar1, tar2 = batch
                inp1 = inp1.to(self.device)
                inp2 = inp2.to(self.device)
                tar1 = tar1.to(self.device)
                tar2 = tar2.to(self.device)
                inp = (inp1, inp2)
                tar = (tar1, tar2)
            elif self.mode == "chain":
                inp, tar1, tar2 = batch
                inp = inp.to(self.device)
                tar1 = tar1.to(self.device)
                tar2 = tar2.to(self.device)
                tar = (tar1, tar2)
            pred = self.forward(inp)
            loss = loss_func(pred, tar) / acc_steps
        scaler.scale(loss).backward()
        if (batch_idx // batch_size + 1) % acc_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        return loss * acc_steps, inp, tar, pred

    def _val_step(self, batch_idx, batch, loss_func):
        if self.mode == "fwd" or self.mode == "bwd":
            inp, tar = batch
            inp = inp.to(self.device)
            tar = tar.to(self.device)
        elif self.mode == "both":
            inp1, inp2, tar1, tar2 = batch
            inp1 = inp1.to(self.device)
            inp2 = inp2.to(self.device)
            tar1 = tar1.to(self.device)
            tar2 = tar2.to(self.device)
            inp = (inp1, inp2)
            tar = (tar1, tar2)
        elif self.mode == "chain":
            inp, tar1, tar2 = batch
            inp = inp.to(self.device)
            tar1 = tar1.to(self.device)
            tar2 = tar2.to(self.device)
            tar = (tar1, tar2)
        pred = self.forward(inp)
        loss = loss_func(pred, tar)
        return loss, inp, tar, pred

    def _on_epoch_end(
        self,
        epoch,
        save_epochs,
        save_path,
        logging,
        loss,
        inp,
        tar,
        pred,
        v_loss,
        v_inp,
        v_tar,
        v_pred,
        val_data,
        rel_err_val,
        chall_err_val,
    ):

        self._print_info()

        if self.mode == "fwd" or self.mode == "bwd":
            logging = logging.append(
                {
                    "loss": loss.item(),
                    "val_loss": v_loss.item(),
                    "rel_l2_error1": l2_error(
                        pred, tar, relative=True, squared=False
                    )[0].item(),
                    "val_rel_l2_error1": l2_error(
                        v_pred, v_tar, relative=True, squared=False
                    )[0].item(),
                },
                ignore_index=True,
                sort=False,
            )
        elif self.mode == "both" or self.mode == "chain":
            logging = logging.append(
                {
                    "loss": loss.item(),
                    "val_loss": v_loss.item(),
                    "rel_l2_error1": l2_error(
                        pred[0], tar[0], relative=True, squared=False
                    )[0].item(),
                    "val_rel_l2_error1": l2_error(
                        v_pred[0], v_tar[0], relative=True, squared=False
                    )[0].item(),
                    "rel_l2_error2": l2_error(
                        pred[1], tar[1], relative=True, squared=False
                    )[0].item(),
                    "val_rel_l2_error2": l2_error(
                        v_pred[1], v_tar[1], relative=True, squared=False
                    )[0].item(),
                },
                ignore_index=True,
                sort=False,
            )
        print(logging.tail(1))

        if (epoch + 1) % save_epochs == 0:
            fig = self._create_figure(
                logging, loss, inp, tar, pred, v_loss, v_inp, v_tar, v_pred
            )

            os.makedirs(save_path, exist_ok=True)
            torch.save(
                self.state_dict(),
                os.path.join(
                    save_path, "model_weights_epoch{}.pt".format(epoch + 1)
                ),
            )
            logging.to_pickle(
                os.path.join(
                    save_path, "losses_epoch{}.pkl".format(epoch + 1)
                ),
            )
            fig.savefig(
                os.path.join(save_path, "plot_epoch{}.png".format(epoch + 1)),
                bbox_inches="tight",
            )

        return logging

    def _create_figure(
        self, logging, loss, inp, tar, pred, v_loss, v_inp, v_tar, v_pred
    ):
        def _implot(sub, im):
            if im.shape[-3] == 2:  # complex image
                p = sub.imshow(
                    torch.sqrt(im.pow(2).sum(-3))[0, :, :].detach().cpu()
                )
            else:  # real image
                p = sub.imshow(im[0, 0, :, :].detach().cpu())
            return p

        if self.mode == "fwd" or self.mode == "bwd":
            fig, subs = plt.subplots(2, 3, clear=True, num=1, figsize=(20, 15))
            v_inp1, v_tar1, v_pred1 = v_inp, v_tar, v_pred
        elif self.mode == "both":
            fig, subs = plt.subplots(2, 5, clear=True, num=1, figsize=(20, 15))
            v_inp1, v_inp2 = v_inp
            v_tar1, v_tar2 = v_tar
            v_pred1, v_pred2 = v_pred
        elif self.mode == "chain":
            fig, subs = plt.subplots(2, 5, clear=True, num=1, figsize=(20, 15))
            v_inp1 = v_inp
            v_tar1, v_tar2 = v_tar
            v_pred1, v_pred2 = v_pred
            v_inp2 = v_inp1

        # training and validation loss
        subs[0, 0].set_title("losses")
        subs[0, 0].semilogy(logging["loss"], label="train")
        subs[0, 0].semilogy(logging["val_loss"], label="val")
        subs[0, 0].legend()

        # validation input
        p01 = _implot(subs[0, 1], v_inp1)
        subs[0, 1].set_title("val inp")
        plt.colorbar(p01, ax=subs[0, 1])

        # validation target
        p11 = _implot(subs[1, 1], v_tar1)
        subs[1, 1].set_title("val tar")
        plt.colorbar(p11, ax=subs[1, 1])

        # validation prediction
        p12 = _implot(subs[1, 2], v_pred1)
        subs[1, 2].set_title(
            "val pred:\n rel. err. = \n "
            "{:1.2e}".format(logging["val_rel_l2_error1"].iloc[-1])
        )
        plt.colorbar(p12, ax=subs[1, 2])

        # validation difference
        p02 = _implot(subs[0, 2], v_pred1 - v_tar1)
        subs[0, 2].set_title("val diff")
        plt.colorbar(p02, ax=subs[0, 2])

        if self.mode == "both" or self.mode == "chain":
            # validation input
            p03 = _implot(subs[0, 3], v_inp2)
            subs[0, 3].set_title("val inp")
            plt.colorbar(p03, ax=subs[0, 3])

            # validation target
            p13 = _implot(subs[1, 3], v_tar2)
            subs[1, 3].set_title("val tar")
            plt.colorbar(p13, ax=subs[1, 3])

            # validation prediction
            p14 = _implot(subs[1, 4], v_pred2)
            subs[1, 4].set_title(
                "val pred:\n rel. err. = \n "
                "{:1.2e}".format(logging["val_rel_l2_error2"].iloc[-1])
            )
            plt.colorbar(p14, ax=subs[1, 4])

            # validation difference
            p04 = _implot(subs[0, 4], v_pred2 - v_tar2)
            subs[0, 4].set_title("val diff")
            plt.colorbar(p04, ax=subs[0, 4])

        return fig


# ----- ----- Iterative Networks ----- -----
class IterativeNet(InvNet):
    def __init__(
        self,
        subnet,
        num_iter,
        lam,
        lam_learnable=True,
        final_dc=True,
        resnet_factor=1.0,
        inverter=None,
        dc_operator=None,
        use_memory=False,
    ):
        super(IterativeNet, self).__init__()
        if isinstance(subnet, list):
            self.subnet = torch.nn.ModuleList(subnet)
        else:
            self.subnet = subnet
        self.num_iter = num_iter
        self.final_dc = final_dc
        self.resnet_factor = resnet_factor
        self.inverter = inverter
        self.dc_operator = dc_operator
        self.use_memory = use_memory
        if not isinstance(lam, (list, tuple)):
            lam = [lam] * num_iter
        if not isinstance(lam_learnable, (list, tuple)):
            lam_learnable = [lam_learnable] * len(lam)

        self.lam = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.tensor(lam[it]), requires_grad=lam_learnable[it]
                )
                for it in range(len(lam))
            ]
        )

    def forward(self, inp):
        x, y = inp  # get sinogram and fbp

        if self.inverter is not None:
            xinv = self.inverter(y)
        else:
            xinv = x

        if self.use_memory is not False:
            x_shape = xinv.shape
            s = torch.zeros(
                x_shape[0],
                self.use_memory,
                x_shape[2],
                x_shape[3],
                device=xinv.device,
            )

        for it in range(self.num_iter):

            if self.use_memory is not False:
                if isinstance(self.subnet, torch.nn.ModuleList):
                    out = self.subnet[it](torch.cat([xinv, s], dim=1))
                else:
                    out = self.subnet(torch.cat([xinv, s], dim=1))
                xinv = self.resnet_factor * xinv + out[:, 0:1, ...]
                s = out[:, 1:, ...]
            else:
                if isinstance(self.subnet, torch.nn.ModuleList):
                    xinv = self.resnet_factor * xinv + self.subnet[it](xinv)
                else:
                    xinv = self.resnet_factor * xinv + self.subnet(xinv)

            if (self.final_dc) or (
                (not self.final_dc) and it < self.num_iter - 1
            ):
                if self.dc_operator is not None:
                    xinv = xinv - self.lam[it] * self.dc_operator((y, xinv))

        return xinv

    def set_learnable_iteration(self, index):
        for i in list(range(self.get_num_iter_max())):
            if i in index:
                self.lam[i].requires_grad = True
                self.subnet[i].unfreeze()
            else:
                self.lam[i].requires_grad = False
                self.subnet[i].freeze()

    def get_num_iter_max(self):
        return len(self.lam)

    def _print_info(self):
        print("Current lambda(s):")
        print(
            [
                self.lam[it].item()
                for it in range(len(self.lam))
                if self.lam[it].numel() == 1
            ]
        )
        print([self.lam[it].requires_grad for it in range(len(self.lam))])
        print("Epoch done", flush=True)


# ----- ----- Data Consistency Layer ----- -----
class DCLsqFPB(torch.nn.Module):
    def __init__(self, operator):
        super(DCLsqFPB, self).__init__()
        self.operator = operator

    def forward(self, inp):
        y, x = inp
        return self.operator.inv(self.operator(x) - y)

    def freeze(self):
        """ Freeze all model weights, i.e. prohibit further updates. """
        for param in self.parameters():
            param.requires_grad = False


# ----- ----- U-Net ----- -----
class GroupUNet(InvNet):
    """ U-Net implementation.

    Based on https://github.com/mateuszbuda/brain-segmentation-pytorch/
    and modified in agreement with their licence:

    -----

    MIT License

    Copyright (c) 2019 mateuszbuda

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_features=32,
        drop_factor=0.0,
        do_center_crop=False,
        num_groups=32,
    ):
        # set properties of UNet
        super(GroupUNet, self).__init__()

        self.do_center_crop = do_center_crop
        kernel_size = 3 if do_center_crop else 2

        self.encoder1 = self._conv_block(
            in_channels,
            base_features,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_1",
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._conv_block(
            base_features,
            base_features * 2,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_2",
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._conv_block(
            base_features * 2,
            base_features * 4,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_3",
        )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._conv_block(
            base_features * 4,
            base_features * 8,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_4",
        )
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._conv_block(
            base_features * 8,
            base_features * 16,
            num_groups,
            drop_factor=drop_factor,
            block_name="bottleneck",
        )

        self.upconv4 = torch.nn.ConvTranspose2d(
            base_features * 16,
            base_features * 8,
            kernel_size=kernel_size,
            stride=2,
        )
        self.decoder4 = self._conv_block(
            base_features * 16,
            base_features * 8,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_4",
        )
        self.upconv3 = torch.nn.ConvTranspose2d(
            base_features * 8,
            base_features * 4,
            kernel_size=kernel_size,
            stride=2,
        )
        self.decoder3 = self._conv_block(
            base_features * 8,
            base_features * 4,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_3",
        )
        self.upconv2 = torch.nn.ConvTranspose2d(
            base_features * 4,
            base_features * 2,
            kernel_size=kernel_size,
            stride=2,
        )
        self.decoder2 = self._conv_block(
            base_features * 4,
            base_features * 2,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_2",
        )
        self.upconv1 = torch.nn.ConvTranspose2d(
            base_features * 2, base_features, kernel_size=kernel_size, stride=2
        )
        self.decoder1 = self._conv_block(
            base_features * 2,
            base_features,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_1",
        )

        self.outconv = torch.nn.Conv2d(
            in_channels=base_features,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x):

        enc1 = self.encoder1(x)

        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = self._center_crop(dec4, enc4.shape[-2], enc4.shape[-1])
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self._center_crop(dec3, enc3.shape[-2], enc3.shape[-1])
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self._center_crop(dec2, enc2.shape[-2], enc2.shape[-1])
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self._center_crop(dec1, enc1.shape[-2], enc1.shape[-1])
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.outconv(dec1)

    def _conv_block(
        self, in_channels, out_channels, num_groups, drop_factor, block_name
    ):
        return torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        block_name + "conv1",
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        block_name + "bn_1",
                        torch.nn.GroupNorm(num_groups, out_channels),
                    ),
                    (block_name + "relu1", torch.nn.ReLU(True)),
                    (block_name + "dr1", torch.nn.Dropout(p=drop_factor)),
                    (
                        block_name + "conv2",
                        torch.nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        block_name + "bn_2",
                        torch.nn.GroupNorm(num_groups, out_channels),
                    ),
                    (block_name + "relu2", torch.nn.ReLU(True)),
                    (block_name + "dr2", torch.nn.Dropout(p=drop_factor)),
                ]
            )
        )

    def _center_crop(self, layer, max_height, max_width):
        if self.do_center_crop:
            _, _, h, w = layer.size()
            xy1 = (w - max_width) // 2
            xy2 = (h - max_height) // 2
            return layer[
                :, :, xy2 : (xy2 + max_height), xy1 : (xy1 + max_width)
            ]
        else:
            return layer


# ----- ----- Tiramisu Network ----- -----
class Tiramisu(InvNet):
    """ Tiramisu network implementation.

    Based on https://github.com/bfortuner/pytorch_tiramisu
    and modified in agreement with their licence:

    -----

    MIT License

    Copyright (c) 2018 Brendan Fortuner

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        drop_factor=0.0,
        down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5),
        pool_factors=(2, 2, 2, 2, 2),
        bottleneck_layers=5,
        growth_rate=8,
        out_chans_first_conv=16,
        use_instance_norm=False,
    ):
        super(Tiramisu, self).__init__()

        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.use_instance_norm = use_instance_norm

        # init counts of channels
        cur_channels_count = 0
        skip_connection_channel_counts = []

        # First Convolution
        self.bn_layer = (
            torch.nn.InstanceNorm2d(out_chans_first_conv)
            if self.use_instance_norm
            else torch.nn.BatchNorm2d(out_chans_first_conv)
        )
        self.add_module(
            "firstconv",
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_chans_first_conv,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )
        cur_channels_count = out_chans_first_conv

        # Downsampling path
        self.denseBlocksDown = torch.nn.ModuleList([])
        self.transDownBlocks = torch.nn.ModuleList([])
        for i in range(len(self.down_blocks)):
            self.denseBlocksDown.append(
                Tiramisu._DenseBlock(
                    cur_channels_count,
                    growth_rate,
                    self.down_blocks[i],
                    drop_factor,
                    use_instance_norm=self.use_instance_norm,
                )
            )
            cur_channels_count += growth_rate * self.down_blocks[i]
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(
                Tiramisu._TransitionDown(
                    cur_channels_count,
                    drop_factor,
                    pool_factors[i],
                    use_instance_norm=self.use_instance_norm,
                )
            )

        # Bottleneck
        self.add_module(
            "bottleneck",
            Tiramisu._Bottleneck(
                cur_channels_count,
                growth_rate,
                bottleneck_layers,
                drop_factor,
                use_instance_norm=self.use_instance_norm,
            ),
        )
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        # Upsampling path
        self.transUpBlocks = torch.nn.ModuleList([])
        self.denseBlocksUp = torch.nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(
                Tiramisu._TransitionUp(
                    prev_block_channels,
                    prev_block_channels,
                    pool_factors[-i - 1],
                )
            )
            cur_channels_count = (
                prev_block_channels + skip_connection_channel_counts[i]
            )

            self.denseBlocksUp.append(
                Tiramisu._DenseBlock(
                    cur_channels_count,
                    growth_rate,
                    up_blocks[i],
                    drop_factor,
                    upsample=True,
                    use_instance_norm=self.use_instance_norm,
                )
            )
            prev_block_channels = growth_rate * self.up_blocks[i]
            cur_channels_count += prev_block_channels

        # Final DenseBlock
        self.transUpBlocks.append(
            Tiramisu._TransitionUp(
                prev_block_channels, prev_block_channels, pool_factors[0]
            )
        )
        cur_channels_count = (
            prev_block_channels + skip_connection_channel_counts[-1]
        )

        self.denseBlocksUp.append(
            Tiramisu._DenseBlock(
                cur_channels_count,
                growth_rate,
                self.up_blocks[-1],
                drop_factor,
                upsample=False,
                use_instance_norm=self.use_instance_norm,
            )
        )
        cur_channels_count += growth_rate * self.up_blocks[-1]

        # Final Conv layer
        self.finalConv = torch.nn.Conv2d(
            in_channels=cur_channels_count,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        out = self.bn_layer(self.firstconv((x)))

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        return out

    # ----- Blocks for Tiramisu -----

    class _DenseLayer(torch.nn.Sequential):
        def __init__(
            self, in_channels, growth_rate, p, use_instance_norm=False
        ):
            super().__init__()
            self.add_module(
                "bn",
                torch.nn.InstanceNorm2d(in_channels)
                if use_instance_norm
                else torch.nn.BatchNorm2d(in_channels),
            )
            self.add_module("relu", torch.nn.ReLU(True))
            self.add_module(
                "conv",
                torch.nn.Conv2d(
                    in_channels,
                    growth_rate,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
            )
            self.add_module("drop", torch.nn.Dropout2d(p=p))

        def forward(self, x):
            return super().forward(x)

    class _DenseBlock(torch.nn.Module):
        def __init__(
            self,
            in_channels,
            growth_rate,
            n_layers,
            p,
            upsample=False,
            use_instance_norm=False,
        ):
            super().__init__()
            self.upsample = upsample
            self.layers = torch.nn.ModuleList(
                [
                    Tiramisu._DenseLayer(
                        in_channels + i * growth_rate,
                        growth_rate,
                        p,
                        use_instance_norm=use_instance_norm,
                    )
                    for i in range(n_layers)
                ]
            )

        def forward(self, x):
            if self.upsample:
                new_features = []
                # we pass all previous activations to each dense layer normally
                # but we only store each layer's output in the new_features
                for layer in self.layers:
                    out = layer(x)
                    x = torch.cat([x, out], dim=1)
                    new_features.append(out)
                return torch.cat(new_features, dim=1)
            else:
                for layer in self.layers:
                    out = layer(x)
                    x = torch.cat([x, out], dim=1)  # 1 = channel axis
                return x

    class _TransitionDown(torch.nn.Sequential):
        def __init__(
            self, in_channels, p, pool_factor, use_instance_norm=False
        ):
            super().__init__()
            self.add_module(
                "bn",
                torch.nn.InstanceNorm2d(in_channels)
                if use_instance_norm
                else torch.nn.BatchNorm2d(in_channels),
            )
            self.add_module("relu", torch.nn.ReLU(inplace=True))
            self.add_module(
                "conv",
                torch.nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            )
            self.add_module("drop", torch.nn.Dropout2d(p))
            self.add_module(
                "maxpool",
                torch.nn.MaxPool2d(
                    kernel_size=pool_factor, stride=pool_factor
                ),
            )

        def forward(self, x):
            return super().forward(x)

    class _TransitionUp(torch.nn.Module):
        def __init__(self, in_channels, out_channels, pool_factor):
            super().__init__()
            self.convTrans = torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=pool_factor,
                padding=0,
                bias=True,
            )

        def forward(self, x, skip):
            out = self.convTrans(x)
            out = Tiramisu._center_crop(out, skip.size(2), skip.size(3))
            out = torch.cat([out, skip], dim=1)
            return out

    class _Bottleneck(torch.nn.Sequential):
        def __init__(
            self,
            in_channels,
            growth_rate,
            n_layers,
            p,
            use_instance_norm=False,
        ):
            super().__init__()
            self.add_module(
                "bottleneck",
                Tiramisu._DenseBlock(
                    in_channels,
                    growth_rate,
                    n_layers,
                    p,
                    upsample=True,
                    use_instance_norm=use_instance_norm,
                ),
            )

        def forward(self, x):
            return super().forward(x)

    def _center_crop(layer, max_height, max_width):
        _, _, h, w = layer.size()
        xy1 = (w - max_width) // 2
        xy2 = (h - max_height) // 2
        return layer[:, :, xy2 : (xy2 + max_height), xy1 : (xy1 + max_width)]


# ----- ----- Dual Domain Networks ----- -----
class DDNet(InvNet):
    """ Learned Primal Dual network implementation.

    Inspired by https://github.com/adler-j/learned_primal_dual.

    Parameters
    ----------
    p_subnet : torch.nn.Module
        Subnetwork operating in the primal (signal) domain. Can be a single
        network (weight sharing between iterations) or a list of networks of
        length `num_iter` (no weight sharing). Set `Ǹone` to use default
        conv nets for each iteration.
    d_subnet : torch.nn.Module
        Subnetwork operating in the dual (measurement) domain. Can be a single
        network (weight sharing between iterations) or a list of networks of
        length `num_iter` (no weight sharing). Set `Ǹone` to use default
        conv nets for each iteration.
    num_iter : int
        Number of primal dual iterations.
    num_mem : int
        Number of additional (memory / hidden state) channels. The respective
        subnetworks need to able to process the extra channels.
    op : LinearOperator
        The forward operator.
    use_inv : bool
        Use pseudo-inverse of the operator instead of the adjoint.
        (Default False)
    use_fbp : bool
        Use the inversion (pseudo-inverse or adjoint) as extra channel in the
        primal domain. (Default False)
    use_bn : bool
        Use a version of batch-normalization (group-norm) in the conv nets that
        are the default subnetworks (has no effect if seperate p_subnet and
        d_subnet are provided).

    """

    def __init__(
        self,
        p_subnet,
        d_subnet,
        num_iter,
        num_mem,
        op,
        use_inv=False,
        use_fbp=False,
        use_bn=False,
    ):
        super(DDNet, self).__init__()

        self.op = op
        self.num_iter = num_iter
        self.num_mem = num_mem
        self.use_inv = use_inv
        self.use_fbp = use_fbp
        self.use_bn = use_bn

        if isinstance(p_subnet, list):
            self.p_subnet = torch.nn.ModuleList(p_subnet)
        elif p_subnet is None:
            extra_channel = 2 if self.use_fbp else 1
            self.p_subnet = torch.nn.ModuleList(
                [
                    DDNet._conv_block(
                        self.num_mem + extra_channel,
                        32,
                        self.num_mem,
                        "p_it_{}".format(it),
                        4,
                        self.use_bn,
                    )
                    for it in range(self.num_iter)
                ]
            )
        else:
            self.p_subnet = p_subnet

        if isinstance(d_subnet, list):
            self.d_subnet = torch.nn.ModuleList(d_subnet)
        elif d_subnet is None:
            self.d_subnet = torch.nn.ModuleList(
                [
                    DDNet._conv_block(
                        self.num_mem + 2,
                        32,
                        self.num_mem,
                        "d_it_{}".format(it),
                        4,
                        self.use_bn,
                    )
                    for it in range(self.num_iter)
                ]
            )
        else:
            self.d_subnet = d_subnet

    def forward(self, inp):

        # get sinogram and fbp
        x, y = inp

        # init primal and dual variables
        primal = torch.cat([torch.zeros_like(x)] * self.num_mem, dim=1)
        dual = torch.cat([torch.zeros_like(y)] * self.num_mem, dim=1)

        adj_or_inv = self.op.inv if self.use_inv else self.op.adj
        fac = 1.0 if self.use_inv else 0.01  # handle bad scaling of adj

        for it in range(self.num_iter):
            # dual variable update (sinogram domain)
            dual_cat = torch.cat(
                [dual, self.op(primal[:, 1:2, ...]), y], dim=1
            )
            if isinstance(self.d_subnet, torch.nn.ModuleList):
                dual_update = self.d_subnet[it](
                    dual_cat
                )  # without weight sharing
            else:
                dual_update = self.d_subnet(dual_cat)  # with weight sharing
            dual = dual + dual_update

            # primal variable update (image domain)
            if self.use_fbp:
                primal_cat = torch.cat(
                    [primal, fac * adj_or_inv(dual[:, 0:1, ...]), x], dim=1
                )
            else:
                primal_cat = torch.cat(
                    [primal, fac * adj_or_inv(dual[:, 0:1, ...])], dim=1
                )
            if isinstance(self.p_subnet, torch.nn.ModuleList):
                primal_update = self.p_subnet[it](
                    primal_cat
                )  # without weight sharing
            else:
                primal_update = self.p_subnet(
                    primal_cat
                )  # with weight sharing
            primal = primal + primal_update

        return primal[:, 0:1, ...]

    @staticmethod
    def _conv_block(
        in_channels,
        inter_channels,
        out_channels,
        block_name,
        num_groups,
        use_bn=False,
    ):
        return torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        block_name + "conv1",
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=inter_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        block_name + "bn_1",
                        torch.nn.GroupNorm(num_groups, inter_channels),
                    )
                    if use_bn
                    else (block_name + "no_bn_1", torch.nn.Identity()),
                    (block_name + "relu1", torch.nn.PReLU()),
                    (
                        block_name + "conv2",
                        torch.nn.Conv2d(
                            in_channels=inter_channels,
                            out_channels=inter_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        block_name + "bn_2",
                        torch.nn.GroupNorm(num_groups, inter_channels),
                    )
                    if use_bn
                    else (block_name + "no_bn_2", torch.nn.Identity()),
                    (block_name + "relu2", torch.nn.PReLU()),
                    (
                        block_name + "conv3",
                        torch.nn.Conv2d(
                            in_channels=inter_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                ]
            )
        )
