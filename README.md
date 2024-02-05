
## AGS-GNN: Attribute-guided Sampling for Graph Neural Network


### Our Cuda C++ implementation of Weighted Neighborhood Sampler

In the folder ```CPPSamplerPy```


### Execution of AGS-GNN

```AGS-NS```


## Installation

### Create or manage virtual environment if Anaconda is available in the system
Check your system if Anaconda module is available. If anaconda is not available install packages in the python base. If anaconda is available, then create a virtual enviroment to manage python packages.  

1. Load Module: ```load module anaconda/version_xxx```
2. Create virtual environment: ```conda create -n agsgnn python=3.7```. Here python version 3.7 is considered.
3. Activate virtual environement: ```conda activate agsgnn``` or ```source activate agsgnn```

Other necessary commands for managing enviroment can be found here : [https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

### Installation of pacakages
The installations are considered for python version 3.7

Most python packages are intalled using ```pip``` or ```conda``` command. For consistency it's better to follow only one of them. If anaconda not available install packages in python base using ```pip``` command.

#### Pytorch

Link of Pytorch installation is here: [https://pytorch.org/](https://pytorch.org/).
If Pytorch is already installed then this is not necessary.

#### Package `ipynb ` for calling one python functions from another Jupyter notebook file

```pip install ipynb```

#### ```PytorchGeometric``` Version and installation

### Libraries Used in Conda Environment

The environment exports are made using following command

```conda list -e > conda_requirements.txt```

can be used to create a conda virtual environment with

```conda create --name <env> --file conda_requirements.txt```

From the conda environment pip packages exported using

```pip list --format=freeze > pip_requirements.txt```

