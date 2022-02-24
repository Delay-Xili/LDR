# LDR — Closed-Loop Data Transcription to an LDR via Minimaxing Rate Reduction

This repository contains the official PyTorch implementation of the paper: 
*[Xili Dai](https://github.com/Delay-Xili), [Shengbang Tong](), Mingyang Li, [Ziyang Li](), [Michael Psenka](), 
[Kwan Ho Ryan Chan](https://ryanchankh.github.io/), Pengyuan Zhai, [Yaodong Yu](https://yaodongyu.github.io/), 
Xiaojun Yuan, Heung Yeung Shum, [Yi Ma](https://people.eecs.berkeley.edu/~yima/). 
["Closed-Loop Data Transcription to an LDR via Minimaxing Rate Reduction."](https://arxiv.org/abs/2111.06636) *.

## Introduction
This work proposes a new computational framework for learning a structured generative model for real-world  datasets. 
In particular we propose to learn a closed-loop transcription between a multi-class multi-dimensional data distribution 
and a linear discriminative representation (LDR) in the feature space that consists of multiple  independent  multi-dimensional linear  subspaces.
This new framework unifies the concepts and benefits of Auto-Encoding and GAN and naturally extends them to the settings of 
learning a both discriminative and generative representation for  multi-class and multi-dimensional real-world data.
Our extensive experiments on many benchmark imagery datasets demonstrate tremendous potential of this new closed-loop formulation: 
under fair comparison, visual quality of the learned decoder and classification performance of the encoder is competitive 
and often better than existing methods based on GAN, VAE, or a combination of both. 
We hope that this repository serves as a reproducible baseline for future researches in this area. 

## Reproducing Results

### Installation for mimicry

For the ease of reproducibility, you are suggested to install miniconda (or anaconda if you prefer) before following executing the following commands.

```bash
git clone https://github.com/Delay-Xili/LDR
cd LDR
conda create -y -n ldr
source activate ldr
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install git+https://github.com/kwotsin/mimicry.git
mkdir data logs
```
Note: we highly encourage to use the version of torch later then 1.10.0 since it brings huge speed up on the calculation of the logdet.

More details of installation can be found [here](https://mimicry.readthedocs.io/en/latest/guides/introduction.html)

### Training

To retrain the neural network from scratch on your own machine, execute the following commands 
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cfg experiments/mnist.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --cfg experiments/cifar10.yaml
CUDA_VISIBLE_DEVICES=0,1 python main.py --cfg experiments/stl10.yaml DATA.ROOT pth/to/the/dataset
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --cfg experiments/CelebA.yaml DATA.ROOT pth/to/the/dataset
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --cfg experiments/LSUN.yaml DATA.ROOT pth/to/the/dataset
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --cfg experiments/ImageNet.yaml DATA.ROOT pth/to/the/dataset
```

Some detail hyper-parameters can be changed directly in corresponding xxx.yaml file. 
We run the experiments on the RTX 3090 with 24GB memories. 
Adjusting the ```CUDA_VISIBLE_DEVICES``` parameter based on your GPU memory.


### Pre-trained Models

You can download our trained models from the following links:

| Datasets | Models      | Results     |
| :------: | :---------: | :---------: |
| MNIST    | mini dcgan  | [link]()    |
| CIFAR-10 | mini dcgan  | [link]()    |
| CIFAR-10 | sngan32     | [link]()    |
| STL-10   | sngan48     | [link]()    |
| CelebA   | sngan128    | [link]()    |
| LSUN     | sngan128    | [link]()    |
| ImageNet | sngan128    | [link]()    |

Each link includes the corresponding results which consist of three files: checkpoints, images, and data. <br>
**checkpoints**: including all saved checkpoint files of G and D during the training.<br>
**images**: including all saved input and reconstructed images during the training.<br>
**data**: including the tensorboard file which recording the changes of loss D, loss G, learning rates of G and D during the training.

### Evaluating the FID and IS score

To evaluate the FID and IS score of your checkpoints under ```checkpoints/```, execute 

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py --cfg experiments/mnist.yaml EVAL.NETD_CKPT path/to/netD/ckpt EVAL.NETG_CKPT path/to/netG/ckpt
CUDA_VISIBLE_DEVICES=0 python evaluation.py --cfg experiments/cifar10.yaml EVAL.NETD_CKPT path/to/netD/ckpt EVAL.NETG_CKPT path/to/netG/ckpt
```

### Testing the classification accuracy

To test the accuracy of your learned discriminator, execute

```bash
CUDA_VISIBLE_DEVICES=0 python test_acc.py --cfg experiments/mnist.yaml --ckpt_epochs 4500 EVAL.DATA_SAMPLE 1000
CUDA_VISIBLE_DEVICES=0 python test_acc.py --cfg experiments/cifar10.yaml --ckpt_epochs 45000 EVAL.DATA_SAMPLE 1000
```

### Disentangled visual attributes as principal components




## Citation

If you find LDR useful in your research, please consider citing:

```
@article{dai2021closed,
  title={Closed-Loop Data Transcription to an LDR via Minimaxing Rate Reduction},
  author={Dai, Xili and Tong, Shengbang and Li, Mingyang and Wu, Ziyang and Chan, Kwan Ho Ryan and Zhai, Pengyuan and Yu, Yaodong and Psenka, Michael and Yuan, Xiaojun and Shum, Heung Yeung and others},
  journal={arXiv preprint arXiv:2111.06636},
  year={2021}
}
```