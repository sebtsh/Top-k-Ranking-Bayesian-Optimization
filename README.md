# Top-k Ranking Bayesian Optimization

The experiments are contained in the experiments folder as both Python scripts and Jupyter notebooks. As of 11 June, only the top-1 of 2 (pairwise comparisons) rankings are included. 

## Setup
The experiments were ran on Ubuntu 18.04 with NVIDIA GPUs with Linux x86_64 driver version >= 418.39 (compatible with CUDA 10.1).
To reproduce the conditions under which the results were obtained:

1. Create a new Anaconda environment with Python 3.6 and activate it
```
conda create --name topkrankingbo python=3.6
conda activate topkrankingbo
```

2. Install TensorFlow 2.1 and TensorFlow Probability 0.9 through pip
```
pip install tensorflow==2.1 tensorflow-probability==0.9
```

3. Install CUDA 10.1, cuDNN 7.6.5, matplotlib and jupyter through conda
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
