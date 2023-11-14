# LapNet

A JAX implementation of the algorithm and calculations described in [Forward Laplacian: A New Computational Framework for Neural Network-based Variational Monte Carlo](https://arxiv.org/abs/2307.08214). 

## Installation

To install LapNet together with all the dependencies, you need to have JAX and LapJAX preinstalled.

JAX can be installed by `pip install jax`. 
To install JAX with CUDA support, use e.g.:

```shell
pip3 install --upgrade jax[cuda]==0.3.24 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Note that the jaxlib version must be compatible to the CUDA version
you use. Please see the
[JAX documentation](https://github.com/google/jax#installation) for more
details.

[LapJAX](https://github.com/YWolfeee/lapjax) is a efficient laplacian computation package described in our paper. It can be installed via:

```
git clone https://github.com/YWolfeee/lapjax.git
pip install ./lapjax
```
Once you have install both packages, go to the `lapnet` directory and run
```shell
pip install -e .
```


## Usage

LapNet uses the `ConfigDict` from
[ml_collections](https://github.com/google/ml_collections) to configure the
system. A few example scripts are included under `lapnet/configs/`. To estimate the ground state of atom C, simply run

```shell
python3 main.py --config lapnet/configs/atom.py --config.system.atom C
```

The system and hyperparameters can be controlled by modifying the config file or
(better, for one-off changes) using flags. Details of all available config settings are
in `lapnet/base_config.py`.

For example, to reproduce the result for the benzene dimer, use the following script:
```
python3 main.py --config lapnet/configs/benzene_dimer/benzene_dimer.py:4.95 --config.pretrain.iterations 50000 --config.pretrain.basis augccpvdz 
```

There are some configuration files that contain multiple configurations. For example, `ferminet_systems_configs.py` contains multiple configurations used in the original ferminet paper. To choose a specific configuration, you need to specify the configuration through `config.system.molecule_name`. For example, to choose the configuration of CH4, one should use:
```
python3 main.py --config lapnet/configs/ferminet_systems.py --config.system.molecule_name CH4
```
For the other config files with multiple configurations, one can check in that configuration file about how to specify different configurations in a command.

The Forward Laplacian option defaults to True. To check the result from original Laplacian calculation method, one can run script:
```
python3 main.py --config lapnet/configs/benzene_dimer/benzene_dimer.py:4.95 --config.pretrain.iterations 50000 --config.pretrain.basis augccpvd --config.optim.forward_laplacian=False
```

See the
[ml_collections](https://github.com/google/ml_collections)' documentation for
further details on the flag syntax. 

Note: to train on large atoms and molecules with large batch sizes, multi-GPU
parallelisation is essential. This is supported via JAX's
[pmap](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap).
Multiple GPUs will be automatically detected and used if available.


## Output

The results directory contains the checkpoints generated during training and `train_stats.csv` which contains the local energy and MCMC acceptance probability for each iteration.


## Citation

If you find this repo useful, please cite our paper:

```
@article{li2023forward,
  title={Forward Laplacian: A New Computational Framework for Neural Network-based Variational Monte Carlo},
  author={Li, Ruichen and Ye, Haotian and Jiang, Du and Wen, Xuelan and Wang, Chuwei and Li, Zhe and Li, Xiang and He, Di and Chen, Ji and Ren, Weiluo and Wang, Liwei},
  journal={arXiv preprint arXiv:2307.08214},
  year={2023}
}
```
