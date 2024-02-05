# CPPsamplerPy
## Setup
In Gilbreth machine, I have activated the following modules.

```
module load gcc/9.3.0
module load cmake/3.15.4
module load cuda/11.7.0
module load cudnn/cuda-11.7_8.6
module load anaconda/2020.11-py38
```

I have created a conda environment named "py310cu117pyg200" and then activated it

```
conda create --name py310cu117pyg200 python=3.10
source activate py310cu117pyg200
```

Also, I have created a jupyter kernel bind to this environment. For this, two packages need to be installed.
```
conda install ipython
conda install ipykernel
```

Then run these two following commands:
```
conda-env-mod kernel -n py310cu117pyg200 --display-name "py310cu117pyg200"
python -m ipykernel install --user --name=py310cu117pyg200 --display-name "py310cu117pyg200"
```

I also installed pytorch and pytorch_geometric packages. Please see the installation instructions for these two package documentation. For pytorch, I have used conda install, but for pyg, I used pip install...

## Compiling and creating the .so file
Once the repo is cloned, execute the following commands inside the project directory. Let project_directory be the full path of the directory.

```
mkdir build
cd build
cmake ..
make
```

It should create a project_directory/build/src/**sampling_module**\*.so file; here * is Python version and architecture-related string. 

## Binding with python codes
In python_codes/Test.ipynb, the sampling_module library is loaded as a package. 

