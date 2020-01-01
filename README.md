# Preferential Bayesian Optimization

## Setup
This package and the notebooks require TensorFlow 2 and GPflow 2. The easiest way to
set up the environment is through the Docker image at sebtsh/gpflow2:

```
docker run --gpus all -p 8888:8888 -v <dir where this repo is cloned>/PBO:/tf/PBO sebtsh/gpflow2
```

This will start a Jupyter notebook server from the Docker image with all the required
dependencies. To take advantage of Nvidia GPUs, use Linux and ensure the Nvidia 
Container Toolkit is installed (follow the instructions at 
https://www.tensorflow.org/install/docker, at GPU Support section).

The experiments can be found in the notebooks folder of this repo.
