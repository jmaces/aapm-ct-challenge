import gzip
import os

import numpy as np
import torch

from config import DATA_PATH


# ----- data transforms -----


class Permute(object):
    """ Permute order of (fbp, sino, target) triples. """

    def __init__(self, perm):
        self.perm = perm

    def __call__(self, inputs):
        out = tuple([inputs[k] for k in self.perm])
        return out


# ----- datasets -----
class CTDataset(torch.utils.data.Dataset):
    """ AAPM Computed Tomography Challenge dataset.

    Loads (fbp, sinogram, target) data from a single data batch file.

    Parameters
    ----------
    subset : string
        One of 'train', 'val', 'test' or valid sub directory path.
        Determines the subf directory to search for data files.
    batch : int
        Number of the data batch to load. One of  [1,2,3,4] for `train` subset
        (Files are split across four files). Should be 1 for `val` and `test`
        subset.
    folds : int
        Number of folds for data splitting (e.g. for cross-validation)
        (Default 10)
    num_fold: int or list
        Number of the current fold to use. One of [0,...,folds-1]. Use a list
        to use multiple folds. (Default 0)
    leave_out : bool
        Leave the specified folds. Otherwise only these folds are kept
        (e.g. set to True for training data and False for
        valdiation data). (Default True)
    transform : callable
        Additional data transforms for pre-processing (Default None)
    device : torch.device
        Device (e.g. CPU, GPU cuda specifier) to place the data on.
        (Default None)

    """

    def __init__(
        self,
        subset,
        batch,
        folds=10,
        num_fold=0,
        leave_out=True,
        transform=None,
        device=None,
    ):
        # choose directory according to subset
        if subset == "train":
            path = os.path.join(DATA_PATH, "training_data")
        elif subset == "val":
            path = os.path.join(DATA_PATH, "validation_data")
        elif subset == "test":
            path = os.path.join(DATA_PATH, "test_data")
        else:
            path = os.path.join(DATA_PATH, subset)

        self.transform = transform
        self.device = device

        # load data files
        self.sinogram = np.load(
            gzip.GzipFile(
                os.path.join(path, "Sinogram_batch{}.npy.gz".format(batch)),
                "r",
            )
        )
        self.fbp = np.load(
            gzip.GzipFile(
                os.path.join(path, "FBP128_batch{}.npy.gz".format(batch)), "r"
            )
        )

        if not subset == "val" and not subset == "test":
            self.phantom = np.load(
                gzip.GzipFile(
                    os.path.join(path, "Phantom_batch{}.npy.gz".format(batch)),
                    "r",
                )
            )
        else:
            self.phantom = 0.0 * self.fbp  # no ground truth data exists here

        assert self.phantom.shape[0] == self.sinogram.shape[0]
        assert self.phantom.shape[0] == self.fbp.shape[0]

        # split dataset for cross validation
        fold_len = self.phantom.shape[0] // folds
        if not isinstance(num_fold, list):
            num_fold = [num_fold]
        p_list, s_list, f_list = [], [], []
        for cur_fold in range(folds):
            il = cur_fold * fold_len
            ir = il + fold_len
            if leave_out ^ (cur_fold in num_fold):
                p_list.append(self.phantom[il:ir])
                s_list.append(self.sinogram[il:ir])
                f_list.append(self.fbp[il:ir])
        self.phantom = np.concatenate(p_list, axis=0)
        self.sinogram = np.concatenate(s_list, axis=0)
        self.fbp = np.concatenate(f_list, axis=0)

        # transform numpy to torch tensor
        self.phantom = torch.tensor(self.phantom, dtype=torch.float)
        self.sinogram = torch.tensor(self.sinogram, dtype=torch.float)
        self.fbp = torch.tensor(self.fbp, dtype=torch.float)

    def __len__(self):
        return self.phantom.shape[0]

    def __getitem__(self, idx):
        # add channel dimension
        out = (
            self.fbp[idx, ...].unsqueeze(0),
            self.sinogram[idx, ...].unsqueeze(0),
            self.phantom[idx, ...].unsqueeze(0),
        )
        # move to device and apply transformations
        if self.device is not None:
            out = tuple([x.to(self.device) for x in out])
        if self.transform is not None:
            out = self.transform(out)
        return out


def load_ct_data(subset, num_batches=4, **kwargs):
    """ Concatenates individual CTDatasets from four files.

    Parameters
    ----------
    subset : string
        one of 'train', 'val', or 'test' or valid sub directory path.
    **kwargs : dictionary
        additional keyword arguments passed on to the CTDatasets.

    Returns
    -------
    Combined dataset from multiple data batch files.
    """

    if not subset == "val" and not subset == "test":
        num_batches = min(num_batches, 4)
    else:
        num_batches = 1

    return torch.utils.data.ConcatDataset(
        [
            CTDataset(subset, batch, **kwargs)
            for batch in range(1, num_batches + 1)
        ]
    )


# ---- run data exploration -----

if __name__ == "__main__":
    # validate data set and print some simple statistics
    tdata = load_ct_data("train", folds=10, num_fold=[0, 9], leave_out=True)
    vdata = load_ct_data("train", folds=10, num_fold=[0, 9], leave_out=False)
    print(len(tdata))
    print(len(vdata))
    y, z, x = tdata[0]
    print(y.shape, z.shape, x.shape)
    print(y.min(), z.min(), x.min())
    print(y.max(), z.max(), x.max())
