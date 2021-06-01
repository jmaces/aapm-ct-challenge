import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm import tqdm

from data_management import load_ct_data
from networks import DCLsqFPB, GroupUNet, IterativeNet, RadonNet


# ----- load configuration -----
import config  # isort:skip

# ----- global configuration -----
mpl.use("agg")
device = torch.device("cuda:0")
torch.cuda.set_device(0)

# ----- network configuration -----
subnet_params = {
    "in_channels": 6,
    "drop_factor": 0.0,
    "base_features": 32,
    "out_channels": 6,
    "num_groups": 32,
}
subnet = GroupUNet

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

dc_operator = DCLsqFPB(operator)
dc_operator.freeze()

it_net_params = {
    "num_iter": 5,
    "lam": 5 * [0.0],
    "lam_learnable": True,
    "final_dc": True,
    "resnet_factor": 1.0,
    "inverter": operator.inv,
    "dc_operator": dc_operator,
    "use_memory": 5,
}

# load networks
net_id = range(0, 10)
it_net = []

for i in net_id:
    subnet_tmp = []
    for j in range(it_net_params["num_iter"]):
        subnet_tmp.append(subnet(**subnet_params).to(device))
    it_net_tmp = IterativeNet(subnet_tmp, **it_net_params).to(device)
    it_net_tmp.load_state_dict(
        torch.load(
            os.path.join(
                config.RESULTS_PATH,
                "ItNet_post_mem_restart"
                "_fixed_lambdas_id{}_train_phase_1".format(i),
                "model_weights_final.pt",
            ),
            map_location=torch.device(device),
        )
    )
    it_net_tmp.freeze()
    it_net_tmp.eval()
    it_net.append(it_net_tmp)

# ensembling
weigths = [1 / 10] * 10


# ----- eval challenge data -----
# always use same folds, num_fold for not train and val
# always use leave_out=True on train and leave_out=False on val data
test_data_params = {
    "folds": 1,
    "num_fold": 0,
    "leave_out": False,
}
test_data = load_ct_data("test", **test_data_params)
data_load_test = torch.utils.data.DataLoader(test_data, 10, shuffle=False)

rec_test_all = []
fbp_test_all = []
sino_test_all = []

# reconstruct
with torch.no_grad():
    for i, v_batch in tqdm(enumerate(data_load_test)):
        rec_test = torch.zeros_like(
            it_net[0]((v_batch[0].to(device), v_batch[1].to(device)))
        )
        for j in range(len(it_net)):
            rec_test += weigths[j] * it_net[j](
                (v_batch[0].to(device), v_batch[1].to(device))
            )
        rec_test_all.append(rec_test)
        fbp_test_all.append(v_batch[0].to(device))
        sino_test_all.append(v_batch[1].to(device))

rec_test_all = torch.cat(rec_test_all, dim=0)
fbp_test_all = torch.cat(fbp_test_all, dim=0)
sino_test_all = torch.cat(sino_test_all, dim=0)

for i in tqdm(range(rec_test_all.shape[0])):
    fig, subs = plt.subplots(1, 4, clear=True, num=1, figsize=(20, 5))

    p0 = subs[0].imshow(sino_test_all[i, 0, :, :].cpu(), aspect="auto")
    subs[0].set_title("sinogram")
    plt.colorbar(p0, ax=subs[0])

    p1 = subs[1].imshow(fbp_test_all[i, 0, :, :].cpu())
    subs[1].set_title("fbp")
    plt.colorbar(p1, ax=subs[1])

    p2 = subs[2].imshow(rec_test_all[i, 0, :, :].cpu())
    subs[2].set_title("rec")
    plt.colorbar(p2, ax=subs[2])

    p3 = subs[3].imshow(
        (
            operator.dot(rec_test_all[i : (i + 1), ...])
            - sino_test_all[i : (i + 1), ...]
        )[0, 0, :, :].cpu(),
        aspect="auto",
    )
    subs[3].set_title("sinogram diff")
    plt.colorbar(p3, ax=subs[3])

    fig.savefig(
        os.path.join(
            config.RESULTS_PATH,
            "test_results",
            "itnet_post_chall_test_{}.png".format(i + 1),
        ),
        bbox_inches="tight",
    )

# save
np.save(
    os.path.join(config.RESULTS_PATH, "test_results", "itnet_post_chall.npy"),
    rec_test_all.squeeze(1).cpu().numpy(),
)
