import os

import matplotlib as mpl
import torch
import torchvision

from data_management import Permute, load_ct_data
from networks import RadonNet


# ----- load configuration -----
import config  # isort:skip

# ----- global configuration -----
mpl.use("agg")
device = torch.device("cuda:0")
torch.cuda.set_device(0)

# ----- network configuration -----
radon_params = {
    "n": [512, 512],
    "n_detect": 1024,
    "angles": torch.linspace(0, 360, 129, requires_grad=False)[:-1],
    "d_source": torch.tensor(1000.00, requires_grad=False),
    "s_detect": torch.tensor(-1.0, requires_grad=False),
    "scale": torch.tensor(0.01, requires_grad=False),
    "flat": True,
    "mode": "fwd",
}
radon_net = RadonNet


# specify block coordinate descent order
def _specify_param(radon_net, train_phase):
    if train_phase % 3 == 0:
        radon_net.OpR.angles.requires_grad = False
        radon_net.OpR.d_source.requires_grad = False
        radon_net.OpR.scale.requires_grad = True
    elif train_phase % 3 == 1:
        radon_net.OpR.angles.requires_grad = False
        radon_net.OpR.d_source.requires_grad = True
        radon_net.OpR.scale.requires_grad = False
    elif train_phase % 3 == 2:
        radon_net.OpR.angles.requires_grad = True
        radon_net.OpR.d_source.requires_grad = False
        radon_net.OpR.scale.requires_grad = False


# ----- training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")


def loss_func(pred, tar):
    return mseloss(pred, tar) / pred.shape[0]


train_phases = 3 * 10
train_params = {
    "num_epochs": int(train_phases / 3) * [3, 2, 1],
    "batch_size": train_phases * [10],
    "loss_func": loss_func,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH,
            "operator_radon_{}_"
            "train_phase_{}".format(
                radon_params["mode"], (i + 1) % (train_phases + 1),
            ),
        )
        for i in range(train_phases + 1)
    ],
    "save_epochs": 1,
    "optimizer": torch.optim.Adam,
    "optimizer_params": int(train_phases / 3)
    * [
        {"lr": 1e-4, "eps": 1e-5},
        {"lr": 1e-0, "eps": 1e-5},
        {"lr": 1e-1, "eps": 1e-5},
    ],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 50, "gamma": 0.75},
    "acc_steps": train_phases * [1],
    "train_transform": torchvision.transforms.Compose([Permute([2, 1])]),
    "val_transform": torchvision.transforms.Compose([Permute([2, 1])]),
    "train_loader_params": {"shuffle": True, "num_workers": 0},
    "val_loader_params": {"shuffle": False, "num_workers": 0},
}

# ----- data configuration -----

# always use same folds, num_fold for noth train and val
# always use leave_out=True on train and leave_out=False on val data
train_data_params = {
    "folds": 400,
    "num_fold": 0,
    "leave_out": True,
}
val_data_params = {
    "folds": 400,
    "num_fold": 0,
    "leave_out": False,
}
train_data = load_ct_data("train", **train_data_params)
val_data = load_ct_data("train", **val_data_params)

# ------ save hyperparameters -------
os.makedirs(train_params["save_path"][-1], exist_ok=True)
with open(
    os.path.join(train_params["save_path"][-1], "hyperparameters.txt"), "w"
) as file:
    for key, value in radon_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in val_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    file.write("train_phases" + ": " + str(train_phases) + "\n")

# ------ construct network and train -----
radon_net = radon_net(**radon_params).to(device)
print(list(radon_net.parameters()))

for i in range(train_phases):
    train_params_cur = {}
    for key, value in train_params.items():
        train_params_cur[key] = (
            value[i] if isinstance(value, (tuple, list)) else value
        )

    print("Phase {}:".format(i + 1))
    for key, value in train_params_cur.items():
        print(key + ": " + str(value))

    _specify_param(radon_net, i)

    radon_net.train_on(train_data, val_data, **train_params_cur)

# ------ bias correction -----

# save biased operator
os.makedirs(train_params["save_path"][-1], exist_ok=True)
torch.save(
    radon_net.state_dict(),
    os.path.join(train_params["save_path"][-1], "model_weights_biased.pt"),
)

# compute and assign mean sinogram error
sino_diff_mean = torch.zeros(128, 1024, device=device)
data_load_train = torch.utils.data.DataLoader(train_data, 10, shuffle=False)
with torch.no_grad():
    for i, v_batch in enumerate(reversed(list(data_load_train))):
        sino_diff_mean += (
            v_batch[1].to(device).mean(dim=0).squeeze()
            - radon_net.OpR(v_batch[0].to(device)).mean(dim=0).squeeze()
        )

    sino_diff_mean = sino_diff_mean / (i + 1)

radon_net.OpR.fwd_offset.data = sino_diff_mean

# save final operator
os.makedirs(train_params["save_path"][-1], exist_ok=True)
torch.save(
    radon_net.state_dict(),
    os.path.join(train_params["save_path"][-1], "model_weights_final.pt"),
)
