
## AGS-GNN: Attribute-guided Sampling for Graph Neural Network

primary libraries used for implementation are `Pytorch-1.9.0`, `PytorchGeometric-2.0.4`, `DeepGraphLibrary (DGL)-latest`, `Apricot-latest`.


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



# Running AGS-GNN

### Dataset

Check file `Submodular/Dataset.ipynb` to see how dataset are loaded.

### Ranking

In the `Submodular` folder we have utitlies function to load dataset, similarity ranking, submodular ranking, and all other utitlies function implementation. These files can be run standalone and you can test on dataset to see how they work.

- `Submodular/KNNWeights.ipynb` for nearest neighbor based feature similarity ranking

- `Submodular/SubmodularWeights.ipynb` for submodular optimization based diversity ranking 


- `Submodular/PretrainedLinkFast.ipynb` is the regression model to train and learn edge weights or learn similarity function



### Samplers: Node Sampler and Graph Samplers

Check the standalone samplers and run it on the dataset to create batches to make sure your system is working properly.

- `Submodular/AGSNodeSampler.ipynb` is for weighted node sampling, we can create multiple batches with different types of samples.

- `Submodular/AGSGraphSampler.ipynb` is for weighted graph sampling, we can create multiple batches with different types of graph sampling.

#### PS: We used Cuda C++ implementation of Weighted Neighborhood Sampler 

We modified the `torch-sparse==0.6.12` for neighborhood sampler. The original random sampler,

```
https://github.com/rusty1s/pytorch_sparse/tree/master/csrc
https://github.com/rusty1s/pytorch_sparse/blob/master/csrc/cpu/neighbor_sample_cpu.cpp
```

In the folder `CPPSamplerPy` we modifed the `torch-sparse` libaries to implement weighted neighborhood sampler and weighted random walk.

The compiled shared library needs to be loaded. You can compile to work in you own enviroment using the `g++` and provided `MakeFile`

In cases if it doesn't work, you can use slightly slower version of these sampler of python implementation by changing the weighted sampler call function in `AGSNodeSampler.ipynb` and `AGSGraphSampler.ipynb`.

For example, in the `AGSNodeSampler.ipynb` you can comment and uncomment these lines to use `C++` or `Python` based implementation.

```
def __call__(self, index: Union[List[int], Tensor]):
        
        output = []
    
        for i,method in enumerate(self.weight_funcs):
            if method == 'random':
                output.append(self.call__original(index)
            else:
                #output.append(self.weighted_sample(index, i)) ##python sparse tensor based implementation
                #output.append(self.call_weighted_sample(index, i)) ## c inspired python implementation
                output.append(self.call_weighted_sample_cpp(index, i)) ## modified c++ installation 
        return output            
```


## Executing Jupyter notebook and Python Script of AGS-GNN

Once you make sure, the Dataset, Ranking Codes, and Samplers are working. In the experiment folder we have implementation of AGS-GNN and other methods. 

Some example methods are,

- `GNNs/AGS-NS.py` will run AGS-GNN with Node Sampling

- `GNNs/AGS-GS-GSAINT.py` and `GNNs/AGS-GS-GSAINT-II.py` will run AGS-GNN with single and dual channel Graph Sampling with underlying GraphSAINT , in the sampling paradigm you can specify, to apply weighted random walk or disjoint graph samping.

-  `GNNs/AGS-NS-Ablation.ipynb` performs single channel, dual channel ablation studies on synthetic and benchmark datasets. In the `Class AGSGNN` change appropriate model and in the sampler settings apply any sampling strategies to create any possible variations.


- `GNNs/AGS-NS-GSAGEorGIN.py` will run AGS-GNN with GraphSAGE or Graph Isomorphic Network (GIN)

-  `GNNs/AGS-NS-GSAGE-CHEB.ipynb` will use dual channel with GraphSAGE for homophily and ChebNet for heterophily


Check the `GNNs` folder for all other detailed implementations.



## Some other optional scripts are:

- `Submodular/EffectiveResistanceWeights.ipynb` can be used to compute effective resistance based edge weights when node features are not avaiable. The sampler will take spectral sparsifier of the graph.

- `Submodular/SpatialConv.ipynb` is the GraphLaplacian inspired message passing convution operation. Based on this, we design `AGS-GCN` with `GCNConv` and `SpatialConv` for dual channel. The model is given at `Submodular/AGSConv.ipynb`





