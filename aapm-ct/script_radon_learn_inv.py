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

d = torch.load(
    os.path.join(
        config.RESULTS_PATH,
        "operator_radon_fwd_train_phase_0",
        "model_weights_final.pt",
    ),
    map_location=device,
)
radon_net = RadonNet.new_from_state_dict(d)
radon_net.to(device)
radon_net.mode = "bwd"
radon_net.OpR.flat = True
radon_net.OpR.filter_type = "hamming"
radon_net.freeze()

# learnable
radon_net.OpR.inv_scale.requires_grad = True

print(list(radon_net.parameters()))

# ----- training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")


def loss_func(pred, tar):
    return mseloss(pred, tar) / pred.shape[0]


train_phases = 1
train_params = {
    "num_epochs": [5],
    "batch_size": [10],
    "loss_func": loss_func,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH,
            "operator_radon_{}_"
            "train_phase_{}".format(
                radon_net.mode, (i + 1) % (train_phases + 1),
            ),
        )
        for i in range(train_phases + 1)
    ],
    "save_epochs": 1,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [{"lr": 1e-3, "eps": 1e-5}],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 1, "gamma": 1.0},
    "acc_steps": [1],
    "train_transform": torchvision.transforms.Compose([Permute([1, 2])]),
    "val_transform": torchvision.transforms.Compose([Permute([1, 2])]),
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
    # for key, value in radon_params.items():
    #     file.write(key + ": " + str(value) + "\n")
    for key, value in train_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in val_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    file.write("train_phases" + ": " + str(train_phases) + "\n")

# ------ construct network and train -----

for i in range(train_phases):
    train_params_cur = {}
    for key, value in train_params.items():
        train_params_cur[key] = (
            value[i] if isinstance(value, (tuple, list)) else value
        )

    print("Phase {}:".format(i + 1))
    for key, value in train_params_cur.items():
        print(key + ": " + str(value))

    radon_net.train_on(train_data, val_data, **train_params_cur)
