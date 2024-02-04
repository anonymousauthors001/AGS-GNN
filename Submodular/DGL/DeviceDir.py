import os
import sys

kernel_name = os.path.basename(sys.executable.replace("/bin/python",""));

if kernel_name == 'py38cu11':
    import ctypes
    ctypes.cdll.LoadLibrary("/apps/gilbreth/cuda-toolkit/cuda-11.2.0/lib64/libcusparse.so.11");
    ctypes.cdll.LoadLibrary("/apps/gilbreth/cuda-toolkit/cuda-11.2.0/lib64/libcublas.so.11");

def get_directory():

    from pathlib import Path
    import os

    if os.uname()[1].find('gilbreth')==0: ##if not darwin(mac/locallaptop)
        DIR='/scratch/gilbreth/das90/Dataset/'
    elif os.uname()[1].find('unimodular')==0:
        DIR='/scratch2/das90/Dataset/'
    elif os.uname()[1].find('Siddharthas')==0:
        DIR='/Users/siddharthashankardas/Purdue/Dataset/'  
    else:
        DIR='./Dataset/'

    Path(DIR).mkdir(parents=True, exist_ok=True)

    RESULTS_DIR=DIR+'RESULTS/'
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
#     print("Data directory: ", DIR)
#     print("Result directory:", RESULTS_DIR)

    return DIR, RESULTS_DIR


def get_device():
    import multiprocessing
    import torch

    NUM_GPUS=0

    try:
        if torch.cuda.is_available():  
            device = torch.device("cuda")
            NUM_GPUS=torch.cuda.device_count()
            print('There are %d GPU(s) available.' % NUM_GPUS)
            print('We will use the GPU:', torch.cuda.get_device_name())# If not...
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")  
    except:
        print('Cuda error using CPU instead.')
        device = torch.device("cpu")  

    print(device)

    # device = torch.device("cpu")  
    # print(device)

    NUM_PROCESSORS=multiprocessing.cpu_count()
    print("Cpu count: ",NUM_PROCESSORS)
    
    return device, NUM_PROCESSORS
    