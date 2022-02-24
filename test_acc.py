import numpy as np
import torch
import argparse
from mcrgan.datasets import get_dataloader
from mcrgan.models import get_models
from mcrgan.default import _C as config
from mcrgan.default import update_config
from utils.utils import sort_dataset, compute_accuracy
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import os
import logging


def nearsub(n_comp, train_features, train_labels, test_features, test_labels):
    """Perform nearest subspace classification.

    Options:
        n_comp (int): number of components for PCA or SVD

    """
    scores_pca = []
    scores_svd = []
    num_classes = train_labels.numpy().max() + 1  # should be correct most of the time
    features_sort, _ = sort_dataset(train_features.numpy(), train_labels.numpy(),
                                          num_classes=num_classes, stack=False)
    fd = features_sort[0].shape[1]
    for j in range(num_classes):
        pca = PCA(n_components=n_comp).fit(features_sort[j])
        pca_subspace = pca.components_.T
        mean = np.mean(features_sort[j], axis=0)
        pca_j = (np.eye(fd) - pca_subspace @ pca_subspace.T) \
                @ (test_features.numpy() - mean).T
        score_pca_j = np.linalg.norm(pca_j, ord=2, axis=0)

        svd = TruncatedSVD(n_components=n_comp).fit(features_sort[j])
        svd_subspace = svd.components_.T
        svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) \
                @ (test_features.numpy()).T
        score_svd_j = np.linalg.norm(svd_j, ord=2, axis=0)

        scores_pca.append(score_pca_j)
        scores_svd.append(score_svd_j)
    test_predict_pca = np.argmin(scores_pca, axis=0)
    test_predict_svd = np.argmin(scores_svd, axis=0)
    acc_pca = compute_accuracy(test_predict_pca, test_labels.numpy())
    acc_svd = compute_accuracy(test_predict_svd, test_labels.numpy())
    print('PCA: {}'.format(acc_pca))
    print('SVD: {}'.format(acc_svd))
    return acc_pca, acc_svd


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='should add the .yaml file',
                        required=True,
                        type=str,
                        )
    parser.add_argument('--ckpt_epochs',
                        help='the # of ckpt be used',
                        required=True,
                        type=int,
                        )
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def get_loader():

    if config.DATA.DATASET == 'mnist':
        transform = transforms.Compose(
            [transforms.Resize(32),
             transforms.ToTensor(),
             transforms.Normalize(0.5, 0.5)])
        trainset = datasets.MNIST(root=config.DATA.ROOT, train=True, download=True, transform=transform)
        testset = datasets.MNIST(root=config.DATA.ROOT, train=False, download=True, transform=transform)

    elif config.DATA.DATASET == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        trainset = datasets.CIFAR10(root=config.DATA.ROOT + '/cifar10', train=True, transform=transform,
                                    download=True)
        testset = datasets.CIFAR10(
            root=config.DATA.ROOT + '/cifar10',
            train=False,
            transform=transform,
            download=True)
    else:
        raise NotImplementedError()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.EVAL.DATA_SAMPLE, shuffle=False, num_workers=config.CUDNN.WORKERS)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.EVAL.DATA_SAMPLE, shuffle=False, num_workers=config.CUDNN.WORKERS)

    return trainloader, testloader


def extract_features(data_loader, encoder, decoder):

    X_all = []
    X_bar_all = []
    Z_all = []
    Z_bar_all = []
    labels_all = []
    train_bar = tqdm(data_loader, desc="extracting all features from dataset")
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        for step, (X, labels) in enumerate(train_bar):
            Z = encoder(X.cuda())
            X_bar = decoder(Z.reshape(Z.shape[0], -1, 1, 1))
            Z_bar = encoder(X_bar.detach())

            X_all.append(X.cpu())
            Z_all.append(Z.view(-1, Z.shape[1]).cpu())
            X_bar_all.append(X_bar.cpu())
            Z_bar_all.append(Z_bar.view(-1, Z_bar.shape[1]).cpu())

            labels_all.append(labels)

    return torch.cat(X_all), torch.cat(Z_all), torch.cat(X_bar_all), torch.cat(Z_bar_all), torch.cat(labels_all)


def test_acc():

    # CUDA_VISIBLE_DEVICES=0 python test_acc.py --cfg pth/to/config.yaml --ckpt_epochs 45000 EVAL.DATA_SAMPLE 1000
    # -----------------------
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    n_comp = 10
    root = os.path.dirname(args.cfg)
    netd_ckpt = f"{root}/checkpoints/netD/netD_{args.ckpt_epochs}_steps.pth"
    netg_ckpt = f"{root}/checkpoints/netG/netG_{args.ckpt_epochs}_steps.pth"

    print(f"loading netD from {netd_ckpt}")
    print(f"loading netG from {netg_ckpt}")

    # The Batch Size here is the amount of data you want to test your FID
    trainloader, testloader = get_loader()

    # Define models and optimizers
    netD, netG = get_models(config.DATA.DATASET, device)

    netD_state_dict, netG_state_dict = torch.load(netd_ckpt), torch.load(netg_ckpt)

    netG.module.load_state_dict(netG_state_dict["model_state_dict"])
    netD.module.load_state_dict(netD_state_dict["model_state_dict"])
    netG.cuda()
    netD.cuda()

    train_X, train_Z, train_X_bar, train_Z_bar, train_labels = extract_features(trainloader, netD, netG)

    test_X, test_Z, test_X_bar, test_Z_bar, test_labels = extract_features(testloader, netD, netG)

    print("----------------------------")
    print("Train_z, Test_z")
    nearsub(n_comp, train_Z, train_labels, test_Z, test_labels)


if __name__ == '__main__':
    test_acc()
    # batch_test_acc()
