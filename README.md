
## AGS-GNN: Attribute-guided Sampling for Graph Neural Network


### Our Cuda C++ implementation of Weighted Neighborhood Sampler 

We modified the `torch-sparse==0.6.12` for neighborhood sampler. The original random sampler,

```
https://github.com/rusty1s/pytorch_sparse/tree/master/csrc
https://github.com/rusty1s/pytorch_sparse/blob/master/csrc/cpu/neighbor_sample_cpu.cpp
```

In the folder `CPPSamplerPy` we modifed the `torch-sparse` libaries to implement weighted neighborhood sampler and weighted random walk.

The compiled shared library needs to be loaded. You can compile to work in you own enviroment using the `g++` and provided `MakeFile`

In cases if it doesn't work, you can use slightly slower version of these sampler of python implementation by changing the weighted sampler call function in `AGSNodeSampler.py` and `AGSGraphSampler.py`.

### Utitlies
In the `Submodular` folder we have utitlies function to load dataset, similarity ranking, submodular ranking, and all other utitlies function implementation


### Execution of AGS-GNN

In the experiment folder we have implementation of AGS-GNN and other methods.

e.g. `AGS-NS-GSAGEorGIN.py`


## Installation

### Create or manage virtual environment if Anaconda is available in the system
Check your system if Anaconda module is available. If anaconda is not available install packages in the python base. If anaconda is available, then create a virtual enviroment to manage python packages.  

1. Load Module: ```load module anaconda/version_xxx```
2. Create virtual environment: ```conda create -n agsgnn python=3.7```. Here python version 3.7 is considered.
3. Activate virtual environement: ```conda activate agsgnn``` or ```source activate agsgnn```

Other necessary commands for managing enviroment can be found here : [https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

### Installation of pacakages
The installations are considered for python version 3.7

Most python packages are intalled using `pip` or `conda` command. For consistency it's better to follow only one of them. If anaconda not available install packages in python base using `pip` command.

#### Pytorch

Link of Pytorch installation is here: [https://pytorch.org/](https://pytorch.org/).
If Pytorch is already installed then this is not necessary.

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Package `ipynb` for calling one python functions from another Jupyter notebook file

```pip install ipynb```

#### `PytorchGeometric` Version 2.0.4 is used and installation


```
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-geometric==2.0.4
```

### Libraries Used in Conda Environment

The environment exports are made using following command

```conda list -e > conda_requirements.txt```

can be used to create a conda virtual environment with

```conda create --name <env> --file conda_requirements.txt```

From the conda environment pip packages exported using

```pip list --format=freeze > pip_requirements.txt```

