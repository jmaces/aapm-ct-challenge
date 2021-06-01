import os
import shutil

import matplotlib as mpl
import torch

from data_management import load_ct_data
from networks import IterativeNet, RadonNet, Tiramisu


# ----- load configuration -----
import config  # isort:skip

if "SGE_TASK_ID" in os.environ:
    job_id = int(os.environ.get("SGE_TASK_ID")) - 1
else:
    job_id = 0

# ----- global configuration -----
mpl.use("agg")
device = torch.device("cuda:0")
torch.cuda.set_device(0)

# ----- network configuration -----
subnet_params = {
    "in_channels": 1,
    "drop_factor": 0.0,
    "out_channels": 1,
    "down_blocks": (4, 5, 7, 10, 12),
    "up_blocks": (12, 10, 7, 5, 4),
    "pool_factors": (2, 2, 2, 2, 2),
    "bottleneck_layers": 16,
    "growth_rate": 18,
    "out_chans_first_conv": 18,
    "use_instance_norm": True,
}
subnet = Tiramisu

# define operator
d = torch.load(
    os.path.join(
        config.RESULTS_PATH,
        "operator_radon_bwd_train_phase_1",
        "model_weights_epoch.pt",
    ),
    map_location=device,
)
radon_net = RadonNet.new_from_state_dict(d)
radon_net.OpR.flat = True
radon_net.OpR.filter_type = "hamming"
radon_net.freeze()
operator = radon_net.OpR.to(device)

it_net_params = {
    "num_iter": 1,
    "lam": 0.0,
    "lam_learnable": False,
    "final_dc": False,
    "resnet_factor": 1.0,
    "inverter": operator.inv,
}

# ----- training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")


def loss_func(pred, tar):
    return mseloss(pred, tar) / pred.shape[0]


train_phases = 1
train_params = {
    "num_epochs": [150],
    "batch_size": [2],
    "loss_func": loss_func,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH,
            "Tiramisu_id{}_"
            "train_phase_{}".format(job_id, (i + 1) % (train_phases + 1)),
        )
        for i in range(train_phases + 1)
    ],
    "save_epochs": 1,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [{"lr": 2e-4, "eps": 1e-5, "weight_decay": 1e-3}],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 1, "gamma": 0.99},
    "acc_steps": [1],
    "train_loader_params": {"shuffle": True, "num_workers": 0},
    "val_loader_params": {"shuffle": False, "num_workers": 0},
}

# ----- data configuration -----

# always use same folds, num_fold for noth train and val
# always use leave_out=True on train and leave_out=False on val data
train_data_params = {
    "folds": 128,
    "num_fold": [
        4 * job_id,
        4 * job_id + 1,
        4 * job_id + 2,
        4 * job_id + 3,
        127,
    ],
    "leave_out": True,
}
val_data_params = {
    "folds": 128,
    "num_fold": [
        4 * job_id,
        4 * job_id + 1,
        4 * job_id + 2,
        4 * job_id + 3,
        127,
    ],
    "leave_out": False,
}
train_data = load_ct_data("train", **train_data_params)
val_data = load_ct_data("train", **val_data_params)

# ------ save hyperparameters -------
os.makedirs(train_params["save_path"][-1], exist_ok=True)
with open(
    os.path.join(train_params["save_path"][-1], "hyperparameters.txt"), "w"
) as file:
    for key, value in subnet_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in it_net_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in val_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    file.write("train_phases" + ": " + str(train_phases) + "\n")

# ------ construct network and train -----
subnet = subnet(**subnet_params).to(device)
it_net = IterativeNet(subnet, **it_net_params).to(device)

for i in range(train_phases):
    train_params_cur = {}
    for key, value in train_params.items():
        train_params_cur[key] = (
            value[i] if isinstance(value, (tuple, list)) else value
        )

    print("Phase {}:".format(i + 1))
    for key, value in train_params_cur.items():
        print(key + ": " + str(value))

    logging = it_net.train_on(train_data, val_data, **train_params_cur)

    epoch = logging["val_chall_err"].argmin() + 1
    it_net.load_state_dict(
        torch.load(
            os.path.join(
                train_params["save_path"][i],
                "model_weights_epoch{}.pt".format(epoch),
            ),
            map_location=torch.device(device),
        )
    )

# ----- pick best weights and save them ----
epoch = logging["val_chall_err"].argmin() + 1

shutil.copyfile(
    os.path.join(
        train_params["save_path"][-2], "model_weights_epoch{}.pt".format(epoch)
    ),
    os.path.join(train_params["save_path"][-2], "model_weights_final.pt"),
)
shutil.copyfile(
    os.path.join(
        train_params["save_path"][-2], "plot_epoch{}.png".format(epoch)
    ),
    os.path.join(
        train_params["save_path"][-2], "plot_epoch_final{}.png".format(epoch)
    ),
)
