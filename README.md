# Top-k Ranking Bayesian Optimization

The experiments are contained in the experiments directory as both Python scripts and Jupyter notebooks. As of 11 June, only the top-1 of 2 (pairwise comparisons) rankings are included. 

## Dependencies
1. Ubuntu >= 18.04
2. NVIDIA GPU(s) with Linux x86_64 driver version >= 418.39 (compatible with CUDA 10.1)
3. Python >= 3.6
4. TensorFlow 2.1
5. TensorFlow Probability 0.9
6. GPflow 2.0.0-rc1
7. CUDA 10.1
8. cuDNN 7.6.5
9. Matplotlib 
9. Jupyter (if accessing notebooks)

## Setup
To reproduce the conditions under which the results were obtained, on Ubuntu 18.04 with NVIDIA GPU(s):

1. Create a new Anaconda environment with Python 3.6 and activate it
```
conda create --name topkrankingbo python=3.6
conda activate topkrankingbo
```

2. Install TensorFlow 2.1 and TensorFlow Probability 0.9 through pip
```
pip install tensorflow==2.1 tensorflow-probability==0.9
```

3. Install CUDA 10.1, cuDNN 7.6.5, Matplotlib and Jupyter through conda
```
conda install -c anaconda cudatoolkit=10.1 cudnn matplotlib jupyter
```

4. Install GPflow 2.0.0-rc1

```
git clone https://github.com/GPflow/GPflow.git
cd GPflow
git tag -l
git checkout tags/2.0.0-rc1
pip install -e .
```

## Running experiments
For the synthetic test functions, the experiments are stored under the directories with the name of the accquisition function (MPES, EI, DTS). For the real world test datasets, the experiments are stored under the directories with the name of the test dataset (CIFAR10, SUSHI).

For example, to run the experiment on the Forrester test function using MPES:
```
python experiments/MPES/MPES_Forr.py
```

## Results
The results of the mean and standard error across all runs of immediate regret at each timestep are stored in the same directory as the experiment notebook/script, in the results directory. For example, to access the results after running the experiment on the Forrester test function using MPES:
```
vim experiments/MPES/results/MPES_Forr/MPES_Forr_mean_sem.txt
```
