# MIMICRY based Experiments

## Installation for mimicry

```bash
conda create -n mimicry python=3.7
conda install -c anaconda tensorflow-gpu==2.6.0
conda install pytorch torchvision
pip install git+https://github.com/kwotsin/mimicry.git
```
Note: in my setting, 
cuda == 10.2
torch == 1.10.0

More details of installation can be found [here](https://mimicry.readthedocs.io/en/latest/guides/introduction.html)


## Training

training on MNIST/CIFAR-10/STL-10/CelebA/LSUN-bedroom/ImageNet

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cfg experiments/xxxx.yaml DATA.DATASET path/to/your/dataset
```

Some detail hyper-parameters can be changed directly in corresponding xxx.yaml file.


## Evaluation

Evaluating 

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py --cfg experiments/dataset.yaml EVAL.NETD_CKPT path/to/netD/ckpt EVAL.NETG_CKPT path/to/netG/ckpt
```