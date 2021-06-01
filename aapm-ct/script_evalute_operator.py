import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_management import load_ct_data
from networks import RadonNet
from operators import l2_error


# ----- load configuration -----
import config  # isort:skip

# ----- global configuration -----
device = torch.device("cuda:0")
torch.cuda.set_device(0)


# define operator
d = torch.load(
    os.path.join(
        config.RESULTS_PATH,
        "operator_radon_bwd_train_phase_1",
        "model_weights.pt",
    ),
    map_location=device,
)
radon_net = RadonNet.new_from_state_dict(d)

# overwrite default settings (if necessary)
radon_net.OpR.flat = True
radon_net.OpR.filter_type = "hamming"

operator = radon_net.OpR.to(device)
print(list(operator.parameters()))

radon_net.freeze()

# ----- data configuration -----

# always use same folds, num_fold for noth train and val
# always use leave_out=True on train and leave_out=False on val data
val_data_params = {
    "folds": 32,
    "num_fold": 0,
    "leave_out": False,
}
val_data = load_ct_data("train", **val_data_params)
data_load_val = torch.utils.data.DataLoader(val_data, 1)


# ----- iterate over val data -----
chall_loss = 0
our_loss = 0
chall_relE = 0
our_relE = 0
sino_relE = 0

with torch.no_grad():
    for i, v_batch in reversed(list(enumerate(data_load_val))):
        # get items
        our_fbp = operator.inv(v_batch[1].to(device))
        chall_fbp = v_batch[0].to(device)
        gt = v_batch[2].to(device)
        sino = v_batch[1].to(device)

        # calc measures
        our_loss += l2_error(our_fbp, gt, relative=False, squared=False)[
            0
        ].item() / np.sqrt(gt.shape[-1] * gt.shape[-2])
        chall_loss += l2_error(chall_fbp, gt, relative=False, squared=False)[
            0
        ].item() / np.sqrt(gt.shape[-1] * gt.shape[-2])
        chall_relE += l2_error(chall_fbp, gt, relative=True, squared=False)[
            0
        ].item()
        our_relE += l2_error(our_fbp, gt, relative=True, squared=False)[
            0
        ].item()
        sino_relE += l2_error(
            operator.dot(gt), sino, relative=True, squared=False
        )[0].item()

    our_loss = our_loss / val_data.__len__()
    chall_loss = chall_loss / val_data.__len__()
    chall_relE = chall_relE / val_data.__len__()
    our_relE = our_relE / val_data.__len__()
    sino_relE = sino_relE / val_data.__len__()


# ----- plotting -----
fig, subs = plt.subplots(2, 5, clear=True, num=1, figsize=(30, 15))


def _implot(sub, im, vmin=0, vmax=0.3, aspect=1.0):
    if im.shape[-3] == 2:  # complex image
        p = sub.imshow(
            torch.sqrt(im.pow(2).sum(-3))[0, :, :].detach().cpu(),
            vmin=vmin,
            vmax=vmax,
            aspect=aspect,
        )
    else:  # real image
        p = sub.imshow(
            im[0, 0, :, :].detach().cpu(), vmin=vmin, vmax=vmax, aspect=aspect
        )
    return p


# gt
p02 = _implot(subs[0, 2], gt)
subs[0, 2].set_title("gt")
plt.colorbar(p02, ax=subs[0, 2])
vmin, vmax = p02.get_clim()

# gt zoom
p12 = _implot(subs[1, 2], gt[..., 300:412, 300:412], vmin=vmin, vmax=vmax)
subs[1, 2].set_title("gt zoom")
plt.colorbar(p12, ax=subs[1, 2])

# our_fbp
p00 = _implot(subs[0, 0], our_fbp, vmin=vmin, vmax=vmax)
subs[0, 0].set_title("our fbp: loss = \n " "{:1.2e}".format(our_loss.item()))
plt.colorbar(p00, ax=subs[0, 0])


# chall_fbp
p10 = _implot(subs[1, 0], chall_fbp, vmin=vmin, vmax=vmax)
subs[1, 0].set_title(
    "chall fbp: loss = \n " "{:1.2e}".format(chall_loss.item())
)
plt.colorbar(p10, ax=subs[1, 0])

# our_fbp zoom
p01 = _implot(subs[0, 1], our_fbp[..., 300:412, 300:412], vmin=vmin, vmax=vmax)
subs[0, 1].set_title("our fbp: relE = \n " "{:1.2e}".format(our_relE))
plt.colorbar(p01, ax=subs[0, 1])


# chall_fbp zoom
p11 = _implot(
    subs[1, 1], chall_fbp[..., 300:412, 300:412], vmin=vmin, vmax=vmax
)
subs[1, 1].set_title("chall fbp: relE = \n " "{:1.2e}".format(chall_relE))
plt.colorbar(p11, ax=subs[1, 1])

# our_fbp difference plot
p03 = _implot(subs[0, 3], (our_fbp - gt), vmin=-0.035, vmax=0.035)
subs[0, 3].set_title("our fbp difference")
plt.colorbar(p03, ax=subs[0, 3])

# chall_fbp difference plot
p04 = _implot(subs[0, 4], (chall_fbp - gt), vmin=-0.035, vmax=0.035)
subs[0, 4].set_title("chall fbp difference")
plt.colorbar(p04, ax=subs[0, 4])

# dc check
p13 = _implot(
    subs[1, 3],
    operator.inv(sino - operator.dot(gt)),
    vmin=-0.0015,
    vmax=0.0015,
)
subs[1, 3].set_title("dc check")
plt.colorbar(p13, ax=subs[1, 3])

# sino difference plot
p14 = _implot(
    subs[1, 4],
    (sino - operator.dot(gt)),
    vmin=-0.0035,
    vmax=0.0035,
    aspect="auto",
)
subs[1, 4].set_title("sino rel err: = \n " "{:1.2e}".format(sino_relE))
plt.colorbar(p14, ax=subs[1, 4])

plt.show()
