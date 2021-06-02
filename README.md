# AAPM DL-Sparse-View CT Challenge Submission - Designing an Iterative Network for Fanbeam-CT with Unknown Geometry

[![GitHub license](https://img.shields.io/github/license/jmaces/aapm-ct-challenge)](https://github.com/jmaces/aapm-ct-challenge/blob/master/LICENSE)
[![code-style black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-pytorch](https://img.shields.io/badge/Made%20with-Pytorch-1f425f.svg)](https://pytorch.org/)

Team: _robust-and-stable_

This repository provides the implementation of our submission to the [AAPM DL-Sparse-View CT Challenge](https://www.aapm.org/GrandChallenge/DL-sparse-view-CT/).

More details can be found in the submission report [Designing an Iterative Network for Fanbeam-CT with Unknown Geometry](http://arxiv.org/abs/2106.00280) by M. Genzel, J. Macdonald, and M. März (2021).


## Usage

The repository contains code to train the complete pipeline (Operator -> UNet -> ItNet -> ItNet-post) of our proposed
reconstruction method, as well as for two comparison networks (Tiramisu & Learned Primal Dual).

_The challenge data is not contained in this repository and needs to be obtained directly from the challenge website._

1. Check (and modify if necessary) the configuration file `config.py`. It specifies the directory paths for the data and results. By default, the data should be stored in the subdirectory `raw_data` and results and model weights are stored in the subdirectory `results`.
2. Identify the forward operator using the scripts named `script_radon_indentify.py` and `script_radon_learn_inv.py`
3. You can check and evaluate the identified operator using the script named `script_evaluate_operator.py`.
4. Train networks using the scripts named `script_train_*.py`.
5. You can evaluate the trained networks on the test data using the scripts named `script_evaluate_test_*.py`


## Requirements

The package versions are the ones we used. Other versions might work as well.

`cudatoolkit` *(v10.1.243)*  
`matplotlib` *(v3.1.3)*  
`numpy` *(v1.18.1)*  
`pandas` *(v1.0.5)*  
`python` *(v3.8.3)*  
`pytorch` *(v1.6.0)*  
`torchvision` *(v0.7.0)*  
`tqdm` *(v4.46.0)*  

## Acknowledgements

Our implementation of the U-Net is based on and adapted from https://github.com/mateuszbuda/brain-segmentation-pytorch/.  
Our implementation of the Tiramisu network is based on and adapted from https://github.com/bfortuner/pytorch_tiramisu/.  
Our implementation of the Learned Primal Dual network is inspired by https://github.com/adler-j/learned_primal_dual/.

Thank you for making your code available.

## License

This repository is MIT licensed, as found in the [LICENSE](LICENSE) file.
