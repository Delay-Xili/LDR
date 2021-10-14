import argparse
import os
import glob
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from generate import gen_testloss, gen_training_accuracy
import train_func as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from sklearn.manifold import TSNE
import random

manualSeed = 1
random.seed(manualSeed)  # python random seed
torch.manual_seed(manualSeed)  # pytorch random seed
np.random.seed(manualSeed)  # numpy random seed
torch.backends.cudnn.deterministic = True

# Generator
class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu, nz, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, nz, 4, 1, 0, bias=False),
            nn.Flatten()
        )

    def forward(self, input):
        return F.normalize(self.main(input))

def sort_dataset(data, labels, num_classes=10, stack=False):
    """Sort dataset based on classes.
    
    Parameters:
        data (np.ndarray): data array
        labels (np.ndarray): one dimensional array of class labels
        num_classes (int): number of classes
        stack (bol): combine sorted data into one numpy array
    
    Return:
        sorted data (np.ndarray), sorted_labels (np.ndarray)

    """
    sorted_data = [[] for _ in range(num_classes)]
    for i, lbl in enumerate(labels):
        sorted_data[lbl].append(data[i])
    sorted_data = [np.stack(class_data) for class_data in sorted_data]
    sorted_labels = [np.repeat(i, (len(sorted_data[i]))) for i in range(num_classes)]
    if stack:
        sorted_data = np.vstack(sorted_data)
        sorted_labels = np.hstack(sorted_labels)
    return sorted_data, sorted_labels

def plot_loss(args):
    """Plot theoretical loss and empirical loss. """
    ## extract loss from csv
    file_dir = os.path.join(args.model_dir, 'losses.csv')
    data = pd.read_csv(file_dir)
    obj_loss_e = -data['loss'].ravel()
    dis_loss_e = data['discrimn_loss_e'].ravel()
    com_loss_e = data['compress_loss_e'].ravel()
    dis_loss_t = data['discrimn_loss_t'].ravel()
    com_loss_t = data['compress_loss_t'].ravel()
    obj_loss_t = dis_loss_t - com_loss_t

    ## Theoretical Loss
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    num_iter = np.arange(len(obj_loss_t))
    ax.plot(num_iter, obj_loss_t, label=r'$\Delta R$', 
                color='green', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, dis_loss_t, label=r'$R$', 
                color='royalblue', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, com_loss_t, label=r'$R^c$', 
                color='coral', linewidth=1.0, alpha=0.8)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_xlabel('Number of iterations', fontsize=10)
    ax.legend(loc='lower right', prop={"size": 15}, ncol=3, framealpha=0.5)
    ax.set_title("Theoretical Loss")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    ## create saving directory
    loss_dir = os.path.join(args.model_dir, 'figures', 'loss')
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
    file_name = os.path.join(loss_dir, 'loss_theoretical.png')
    plt.savefig(file_name, dpi=400)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(loss_dir, 'loss_theoretical.pdf')
    plt.savefig(file_name, dpi=400)
    plt.close()
    print("Plot saved to: {}".format(file_name))

    ## Empirial Loss
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    num_iter = np.arange(len(obj_loss_e))
    ax.plot(num_iter, obj_loss_e, label=r'$\Delta R$', 
                color='green', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, dis_loss_e, label=r'$R$', 
                color='royalblue', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, com_loss_e, label=r'$R^c$', 
                color='coral', linewidth=1.0, alpha=0.8)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_xlabel('Number of iterations', fontsize=10)
    ax.legend(loc='lower right', prop={"size": 15}, ncol=3, framealpha=0.5)
    ax.set_title("Empirical Loss")
    plt.tight_layout()
    file_name = os.path.join(loss_dir, 'loss_empirical.png')
    plt.savefig(file_name, dpi=400)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(loss_dir, 'loss_empirical.pdf')
    plt.savefig(file_name, dpi=400)
    plt.close()
    print("Plot saved to: {}".format(file_name))


def plot_loss_log(args):
    """Plot theoretical log loss. """
    def moving_average(arr, size=(9, 9)):
        assert len(size) == 2
        mean_ = []
        min_ = []
        max_ = [] 
        for i in range(len(arr)):
            l, r = i-size[0], i+size[1]
            l, r = np.max([l, 0]), r + 1 #adjust bounds
            mean_.append(np.mean(arr[l:r]))
            min_.append(np.amin(arr[l:r]))
            max_.append(np.amax(arr[l:r]))
        return mean_, min_, max_

    ## extract loss from csv
    file_dir = os.path.join(args.model_dir, 'losses.csv')
    data = pd.read_csv(file_dir)
    dis_loss_e = data['discrimn_loss_e'].ravel()
    com_loss_e = data['compress_loss_e'].ravel()
    obj_loss_e = -data['loss'].ravel()

    avg_dis_loss_e, min_dis_loss_e, max_dis_loss_e = moving_average(dis_loss_e)
    avg_com_loss_e, min_com_loss_e, max_com_loss_e = moving_average(com_loss_e)
    avg_obj_loss_e, min_obj_loss_e, max_obj_loss_e = moving_average(obj_loss_e)

    ## Empirical Loss
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    num_iter = np.arange(1, len(obj_loss_e))
    ax.plot(np.log(num_iter), avg_obj_loss_e[:-1], label=r'$\Delta R$', 
                color='green', linewidth=1.0, alpha=0.8)
    ax.plot(np.log(num_iter), avg_dis_loss_e[:-1], label=r'$R$', 
                color='royalblue', linewidth=1.0, alpha=0.8)
    ax.plot(np.log(num_iter), avg_com_loss_e[:-1], label=r'$R^c$', 
                color='coral', linewidth=1.0, alpha=0.8)
    # ax.fill_between(np.log(num_iter), max_obj_loss_t[:-1], min_obj_loss_t[:-1], facecolor='green', alpha=0.5)
    # ax.fill_between(np.log(num_iter), max_dis_loss_t[:-1], min_dis_loss_t[:-1], facecolor='royalblue', alpha=0.5)
    # ax.fill_between(np.log(num_iter), max_com_loss_t[:-1], min_com_loss_t[:-1], facecolor='coral', alpha=0.5)
    ax.vlines(4, ymin=0, ymax=80, linestyle="--", linewidth=1.0, color='gray', alpha=0.8)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_xlabel('Number of iterations ($\log_2$ scale)', fontsize=14)
    ax.legend(loc='lower right', prop={"size": 14}, ncol=3, framealpha=0.5)
    [tick.label.set_fontsize(14) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
    plt.tight_layout()

    # save
    loss_dir = os.path.join(args.model_dir, "figures", "loss_log")
    if not os.path.exists(loss_dir):
        os.mkdir(loss_dir)
    file_name = os.path.join(loss_dir, 'loss_empirical.png')
    plt.savefig(file_name, dpi=400)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(loss_dir, 'loss_empirical.pdf')
    plt.savefig(file_name, dpi=400)
    print("Plot saved to: {}".format(file_name))
    plt.close()


def plot_loss_layer(args):
    """Plot loss per layer. """
    ## create saving directory
    loss_dir = os.path.join(args.model_dir, 'figures', 'loss')
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    layer_dir = os.path.join(args.model_dir, "layers")
    for l, filename in enumerate(os.listdir(layer_dir)):
        data = pd.read_csv(os.path.join(layer_dir, filename))

        ## extract loss from csv
        obj_loss_e = -data['loss'].ravel()
        dis_loss_e = data['discrimn_loss_e'].ravel()
        com_loss_e = data['compress_loss_e'].ravel()
        dis_loss_t = data['discrimn_loss_t'].ravel()
        com_loss_t = data['compress_loss_t'].ravel()
        obj_loss_t = dis_loss_t - com_loss_t

        ## Theoretical Loss
        fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True, dpi=400)
        num_iter = np.arange(len(obj_loss_t))
        ax.plot(num_iter, obj_loss_t, label=r'$\mathcal{L}^d-\mathcal{L}^c$', 
                    color='green', linewidth=1.0, alpha=0.8)
        ax.plot(num_iter, dis_loss_t, label=r'$\mathcal{L}^d$', 
                    color='royalblue', linewidth=1.0, alpha=0.8)
        ax.plot(num_iter, com_loss_t, label=r'$\mathcal{L}^c$', 
                    color='coral', linewidth=1.0, alpha=0.8)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_xlabel('Number of iterations', fontsize=10)
        ax.legend(loc='lower right', prop={"size": 15}, ncol=3, framealpha=0.5)
        ax.set_title("Theoretical Loss")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        file_name = os.path.join(loss_dir, f'layer{l}_loss_theoretical.png')
        plt.savefig(file_name, dpi=400)
        print("Plot saved to: {}".format(file_name))
        # file_name = os.path.join(loss_dir, f'layer{l}_loss_theoretical.pdf')
        # plt.savefig(file_name, dpi=400)
        plt.close()
        # print("Plot saved to: {}".format(file_name))

        ## Empirial Loss
        fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True, dpi=400)
        num_iter = np.arange(len(obj_loss_e))
        ax.plot(num_iter, obj_loss_e, label=r'$\widehat{\mathcal{L}^d}-\widehat{\mathcal{L}^c}$', 
                    color='green', linewidth=1.0, alpha=0.8)
        ax.plot(num_iter, dis_loss_e, label=r'$\widehat{\mathcal{L}^d}$', 
                    color='royalblue', linewidth=1.0, alpha=0.8)
        ax.plot(num_iter, com_loss_e, label=r'$\widehat{\mathcal{L}^c}$', 
                    color='coral', linewidth=1.0, alpha=0.8)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_xlabel('Number of iterations', fontsize=10)
        ax.legend(loc='lower right', prop={"size": 15}, ncol=3, framealpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title("Empirical Loss")
        plt.tight_layout()
        # file_name = os.path.join(loss_dir, f'layer{l}_loss_empirical.png')
        # plt.savefig(file_name, dpi=400)
        # print("Plot saved to: {}".format(file_name))
        # file_name = os.path.join(loss_dir, f'layer{l}_loss_empirical.pdf')
        # plt.savefig(file_name, dpi=400)
        plt.close()
        # print("Plot saved to: {}".format(file_name))


def plot_pca(args, features, labels, epoch, select_label=None):
    """Plot PCA of learned features. """
    ## create save folder
    pca_dir = os.path.join(args.model_dir, 'figures', 'pca')
    if not os.path.exists(pca_dir):
        os.makedirs(pca_dir)

    ## perform PCA on features
    n_comp = np.min([args.comp, features.shape[1]])
    num_classes = labels.numpy().max() + 1
    features_sort, _ = sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    pca = PCA(n_components=n_comp).fit(features.numpy())
    sig_vals = [pca.singular_values_]
    sig_vals_each_class = []
    components_each_class = []
    means_each_class = []
    for c in range(num_classes): 
        pca = PCA(n_components=n_comp).fit(features_sort[c])
        sig_vals.append((pca.singular_values_))
        sig_vals_each_class.append((pca.singular_values_))
        components_each_class.append((pca.components_))
        means_each_class.append((pca.mean_))
        print(sig_vals_each_class, components_each_class, means_each_class)
    ## plot features
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5), dpi=500)
    x_min = np.min([len(sig_val) for sig_val in sig_vals])
    ax.plot(np.arange(x_min), sig_vals[0][:x_min], '-p', markersize=3, markeredgecolor='black',
        linewidth=1.5, color='tomato')
    map_vir = plt.cm.get_cmap('Blues', 6)
    norm = plt.Normalize(-10, 10)
    class_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    norm_class = norm(class_list)
    color = map_vir(norm_class)
    for c, sig_val in enumerate(sig_vals[1:]):
        if select_label is not None:
            color_c = 'green' if c==select_label else color[c]
        else:
            color_c = color[c]
        # color_c = 'green' if c<5 else color[c]
        ax.plot(np.arange(x_min), sig_val[:x_min], '-o', markersize=3, markeredgecolor='black',
                alpha=0.6, linewidth=1.0, color=color_c)
    ax.set_xticks(np.arange(0, x_min, 5))
    ax.set_yticks(np.arange(0, 35, 5))
    ax.set_xlabel("components", fontsize=14)
    ax.set_ylabel("sigular values", fontsize=14)
    [tick.label.set_fontsize(12) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(12) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()
    
    # save statistics
    np.save(os.path.join(pca_dir, f"sig_vals_epo{epoch}.npy"), sig_vals)
    np.save(os.path.join(pca_dir, f"sig_vals_each_class_epo{epoch}.npy"), sig_vals_each_class)
    np.save(os.path.join(pca_dir, f"components_each_class_epo{epoch}.npy"), components_each_class)
    np.save(os.path.join(pca_dir, f"means_each_class_epo{epoch}.npy"), means_each_class)

    file_name = os.path.join(pca_dir, f"pca_classVclass_epoch{epoch}.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(pca_dir, f"pca_classVclass_epoch{epoch}.pdf")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()


def plot_hist(args, features, labels, epoch):
    """Plot histogram of class vs. class. """
    ## create save folder
    hist_folder = os.path.join(args.model_dir, 'figures', 'hist')
    if not os.path.exists(hist_folder):
        os.makedirs(hist_folder)

    num_classes = labels.numpy().max() + 1
    features_sort, _ = sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    for i in range(num_classes):
        for j in range(i, num_classes):
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5), dpi=250)
            if i == j:
                sim_mat = features_sort[i] @ features_sort[j].T
                sim_mat = sim_mat[np.triu_indices(sim_mat.shape[0], k = 1)]
            else:
                sim_mat = (features_sort[i] @ features_sort[j].T).reshape(-1)
            ax.hist(sim_mat, bins=40, color='red', alpha=0.5)
            ax.set_xlabel("cosine similarity")
            ax.set_ylabel("count")
            ax.set_title(f"Class {i} vs. Class {j}")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            fig.tight_layout()

            file_name = os.path.join(hist_folder, f"hist_{i}v{j}")
            fig.savefig(file_name)
            plt.close()
            print("Plot saved to: {}".format(file_name))


def plot_traintest(args, path_test):
    """Plot traintest loss. """
    def process_df(data):
        epochs = data['epoch'].ravel().max()
        mean_, max_, min_ = [], [], []
        for epoch in np.arange(epochs+1):
            row = data[data['epoch'] == epoch].drop(columns=['step', 'discrimn_loss_e', 'compress_loss_e'])
            mean_.append(row.mean())
            max_.append(row.max())
            min_.append(row.min())
        return pd.DataFrame(mean_), pd.DataFrame(max_), pd.DataFrame(min_)

    def moving_average(arr, size=(9, 9)):
        assert len(size) == 2
        mean_ = []
        min_ = []
        max_ = [] 
        for i in range(len(arr)):
            l, r = i-size[0], i+size[1]
            l, r = np.max([l, 0]), r + 1 #adjust bounds
            mean_.append(np.mean(arr[l:r]))
            min_.append(np.amin(arr[l:r]))
            max_.append(np.amax(arr[l:r]))
        return mean_, min_, max_

    path_train = os.path.join(args.model_dir, 'losses.csv')
    path_test = os.path.join(args.model_dir, 'losses_test.csv')
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)
    df_train_mean, df_train_max, df_train_min = process_df(df_train)
    df_test_mean, df_test_max, df_test_min = process_df(df_test)

    train_dis_loss_mean = df_train_mean['discrimn_loss_t'].ravel()
    train_com_loss_mean = df_train_mean['compress_loss_t'].ravel()
    train_obj_loss_mean = train_dis_loss_mean - train_com_loss_mean
    train_dis_loss_max = df_train_max['discrimn_loss_t'].ravel()
    train_com_loss_max = df_train_max['compress_loss_t'].ravel()
    train_obj_loss_max = train_dis_loss_max - train_com_loss_max
    train_dis_loss_min = df_train_min['discrimn_loss_t'].ravel()
    train_com_loss_min = df_train_min['compress_loss_t'].ravel()
    train_obj_loss_min = train_dis_loss_min - train_com_loss_min

    test_dis_loss_mean = df_test_mean['discrimn_loss_t'].ravel()
    test_com_loss_mean = df_test_mean['compress_loss_t'].ravel()
    test_obj_loss_mean = test_dis_loss_mean - test_com_loss_mean
    test_dis_loss_max = df_test_max['discrimn_loss_t'].ravel()
    test_com_loss_max = df_test_max['compress_loss_t'].ravel()
    test_obj_loss_max = test_dis_loss_max - test_com_loss_max
    test_dis_loss_min = df_test_min['discrimn_loss_t'].ravel()
    test_com_loss_min = df_test_min['compress_loss_t'].ravel()
    test_obj_loss_min = test_dis_loss_min - test_com_loss_min

    train_obj_loss_mean = moving_average(train_obj_loss_mean)[0] 
    test_obj_loss_mean = moving_average(test_obj_loss_mean)[0]
    train_dis_loss_mean = moving_average(train_dis_loss_mean)[0]
    test_dis_loss_mean = moving_average(test_dis_loss_mean)[0]
    train_com_loss_mean = moving_average(train_com_loss_mean)[0]
    test_com_loss_mean = moving_average(test_com_loss_mean)[0]
    plt.rc('text', usetex=True)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] #+ plt.rcParams['font.serif']
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    num_iter = np.arange(len(train_obj_loss_mean))
    ax.plot(num_iter, train_obj_loss_mean, label=r'$\Delta R$ (train)', 
                color='green', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, test_obj_loss_mean, label='$\Delta R$ (test)', 
                color='green', linewidth=1.0, alpha=0.8, linestyle='--')
    ax.plot(num_iter, train_dis_loss_mean, label='$R$ (train)', 
                color='royalblue', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, test_dis_loss_mean, label='$R$ (test)', 
                color='royalblue', linewidth=1.0, alpha=0.8, linestyle='--')
    ax.plot(num_iter, train_com_loss_mean, label='$R^c$ (train)', 
                color='coral', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, test_com_loss_mean, label='$R^c$ (test)', 
                color='coral', linewidth=1.0, alpha=0.8, linestyle='--')
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.legend(loc='lower right', frameon=True, fancybox=True, prop={"size": 12}, ncol=3, framealpha=0.5)
    ax.set_ylim(0, 80)
    [tick.label.set_fontsize(14) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()

    save_dir = os.path.join(args.model_dir, 'figures', "traintest")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"loss_traintest.png")
    fig.savefig(file_name, dpi=400)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(save_dir, f"loss_traintest.pdf")
    fig.savefig(file_name, dpi=400)
    print("Plot saved to: {}".format(file_name))
    plt.close()
    

def plot_nearest_component_supervised(args, features, labels, epoch, trainset):
    """Find corresponding images to the nearests component. """
    ## perform PCA on features
    features_sort, _ = sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=10, stack=False)
    data_sort, _ = sort_dataset(trainset.data, labels.numpy(), 
                            num_classes=10, stack=False)
    nearest_data = []
    for c in range(10):
        pca = TruncatedSVD(n_components=10, random_state=10).fit(features_sort[c])
        proj = features_sort[c] @ pca.components_.T
        img_idx = np.argmax(np.abs(proj), axis=0)
        nearest_data.append(np.array(data_sort[c])[img_idx])
    
    fig, ax = plt.subplots(ncols=10, nrows=10, figsize=(10, 10))
    for r in range(10):
        for c in range(10):
            ax[r, c].imshow(nearest_data[r][c])
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            ax[r, c].spines['top'].set_visible(False)
            ax[r, c].spines['right'].set_visible(False)
            ax[r, c].spines['bottom'].set_linewidth(False)
            ax[r, c].spines['left'].set_linewidth(False)
            if c == 0:
                ax[r, c].set_ylabel(f"comp {r}")
    ## save
    save_dir = os.path.join(args.model_dir, 'figures', 'nearcomp_sup')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"nearest_data.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(save_dir, f"nearest_data.pdf")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()


def plot_nearest_component_unsupervised(args, features, labels, epoch, trainset):
    """Find corresponding images to the nearests component. """
    save_dir = os.path.join(args.model_dir, 'figures', 'nearcomp_unsup')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    feature_dim = features.shape[1]
    pca = TruncatedSVD(n_components=feature_dim-1, random_state=10).fit(features)
    for j, comp in enumerate(pca.components_):
        proj = (features @ comp.T).numpy()
        img_idx = np.argsort(np.abs(proj), axis=0)[::-1][:10]
        nearest_vals = proj[img_idx]
        print(img_idx, trainset.data.shape)
        nearest_data = trainset.data[img_idx.copy()]
        fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(5, 2))
        i = 0
        for r in range(2):
            for c in range(5):
                ax[r, c].imshow(nearest_data[i])
                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])
                ax[r, c].spines['top'].set_visible(False)
                ax[r, c].spines['right'].set_visible(False)
                ax[r, c].spines['bottom'].set_linewidth(False)
                ax[r, c].spines['left'].set_linewidth(False)
                i+= 1
        file_name = os.path.join(save_dir, f"nearest_comp{j}.png")
        fig.savefig(file_name)
        print("Plot saved to: {}".format(file_name))
        plt.close()


def plot_nearest_component_class(args, features, labels, epoch, trainset):
    """Find corresponding images to the nearests component per class. """
    features_sort, _ = sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=10, stack=False)
    data_sort, _ = sort_dataset(trainset.data, labels.numpy(), 
                            num_classes=10, stack=False)

    for class_ in range(10):
        nearest_data = []
        nearest_val = []
        pca = TruncatedSVD(n_components=10, random_state=10).fit(features_sort[class_])
        for j in range(8):
            proj = features_sort[class_] @ pca.components_.T[:, j]
            img_idx = np.argsort(np.abs(proj), axis=0)[::-1][:10]
            nearest_val.append(proj[img_idx])
            nearest_data.append(np.array(data_sort[class_])[img_idx])
        
        fig, ax = plt.subplots(ncols=10, nrows=8, figsize=(10, 10))
        for r in range(8):
            for c in range(10):
                ax[r, c].imshow(nearest_data[r][c])
                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])
                ax[r, c].spines['top'].set_visible(False)
                ax[r, c].spines['right'].set_visible(False)
                ax[r, c].spines['bottom'].set_linewidth(False)
                ax[r, c].spines['left'].set_linewidth(False)
                ax[r, c].set_xlabel(f"proj: {nearest_val[r][c]:.2f}")
                if c == 0:
                    ax[r, c].set_ylabel(f"comp {r}")
        fig.tight_layout()

        ## save
        save_dir = os.path.join(args.model_dir, 'figures', 'nearcomp_class')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f"nearest_class{class_}.png")
        fig.savefig(file_name)
        print("Plot saved to: {}".format(file_name))
        file_name = os.path.join(save_dir, f"nearest_class{class_}.pdf")
        fig.savefig(file_name)
        print("Plot saved to: {}".format(file_name))
        plt.close()


def plot_accuracy(args, path):
    """Plot train and test accuracy. """
    def moving_average(arr, size=(9, 9)):
        assert len(size) == 2
        mean_ = []
        min_ = []
        max_ = [] 
        for i in range(len(arr)):
            l, r = i-size[0], i+size[1]
            l, r = np.max([l, 0]), r + 1 #adjust bounds
            mean_.append(np.mean(arr[l:r]))
            min_.append(np.amin(arr[l:r]))
            max_.append(np.amax(arr[l:r]))
        return mean_, min_, max_
    df = pd.read_csv(path)
    acc_train = df['acc_train'].ravel()
    acc_test = df['acc_test'].ravel()
    epochs = np.arange(len(df))

    acc_train, _, _ = moving_average(acc_train)
    acc_test, _, _ = moving_average(acc_test)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=400)
    ax.plot(epochs, acc_train, label='train', alpha=0.6, color='lightcoral')
    ax.plot(epochs, acc_test, label='test', alpha=0.6, color='cornflowerblue')
    ax.legend(loc='lower right', frameon=True, fancybox=True, prop={"size": 14}, ncol=2, framealpha=0.5)
    ax.set_xlabel("epochs", fontsize=14)
    ax.set_ylabel("accuracy", fontsize=14)
    [tick.label.set_fontsize(14) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()

    ## save
    save_dir = os.path.join(args.model_dir, 'figures', 'acc')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"acc_traintest.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(save_dir, f"acc_traintest.pdf")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()


def plot_heatmap(args, features, labels, epoch):
    """Plot heatmap of cosine simliarity for all features. """
    num_classes = labels.numpy().max() + 1
    features_sort, _ = sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    for i in range(len(features_sort)):
        features_sort[i] = features_sort[i]#[:2000]
    features_sort_ = np.vstack(features_sort)
    sim_mat = np.abs(features_sort_ @ features_sort_.T)

    plt.rc('text', usetex=False)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] #+ plt.rcParams['font.serif']

    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    im = ax.imshow(sim_mat, cmap='Blues')
    # im = ax.imshow(sim_mat, cmap='bwr')
    fig.colorbar(im, pad=0.02, drawedges=0, ticks=[0, 0.5, 1])
    ax.set_xticks(np.linspace(0, 50000, 6))
    ax.set_yticks(np.linspace(0, 50000, 6))
    [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()

    
    save_dir = os.path.join(args.model_dir, 'figures', 'heatmaps')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"heatmat_epoch{epoch}.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(save_dir, f"heatmat_epoch{epoch}.pdf")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()


def plot_heatmap_ZnZ_hat(args, features, features_rec, labels, epoch):
    """Plot heatmap of cosine simliarity for all features. """
    num_classes = labels.numpy().max() + 1
    features_sort, _ = sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    features_rec_sort, _ = sort_dataset(features_rec.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    for i in range(len(features_sort)):
        features_sort[i] = features_sort[i]#[:2000]
        features_rec_sort[i] = features_rec_sort[i]
    features_sort_ = np.vstack(features_sort)
    features_rec_sort_ = np.vstack(features_rec_sort)

    sim_mat = np.abs(features_sort_ @ features_rec_sort_.T)

    plt.rc('text', usetex=False)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] #+ plt.rcParams['font.serif']

    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    im = ax.imshow(sim_mat, cmap='Blues')
    # im = ax.imshow(sim_mat, cmap='bwr')
    fig.colorbar(im, pad=0.02, drawedges=0, ticks=[0, 0.5, 1])
    ax.set_xticks(np.linspace(0, 50000, 6))
    ax.set_yticks(np.linspace(0, 50000, 6))
    [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()

    
    save_dir = os.path.join(args.model_dir, 'figures', 'heatmaps')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"ZnZ_hat_heatmat_epoch{epoch}.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(save_dir, f"ZnZ_hat_heatmat_epoch{epoch}.pdf")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

def plot_tsne(args, features, labels, epoch):
    """Plot tsne of features. """
    num_classes = labels.numpy().max() + 1
    features_sort, labels_sort = sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    for i in range(len(features_sort)):
        features_sort[i] = features_sort[i][:200]
        labels_sort[i] = labels_sort[i][:200]
    features_sort_ = np.vstack(features_sort)
    labels_sort_ = np.hstack(labels_sort)
    print(features_sort_.shape, labels_sort_.shape)

    print('TSNEing')
    feature_tsne=TSNE(n_components=2, init='pca').fit_transform(features_sort_)
    print('TSNE Finished')

    #plot
    color = plt.cm.rainbow(np.linspace(0, 1, 10))

    plt.figure(figsize=(9,7))
    for i in range(10):
        plt.scatter(feature_tsne[labels_sort_==i,0],feature_tsne[labels_sort_==i,1],marker='o',s=30,edgecolors='k',linewidths=1,c=color[i])

    plt.xticks([])
    plt.yticks([])

    save_dir = os.path.join(args.model_dir, 'figures', 'tsne')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"tsne_train_feature_epoch{epoch}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

def plot_tsne_all(args, features, features_recon, labels, epoch):
    """Plot tsne of features. """
    num_classes = labels.numpy().max() + 1
    features_sort, labels_sort = sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    features_recon_sort, labels_sort = sort_dataset(features_recon.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    for i in range(len(features_sort)):
        features_sort[i] = features_sort[i][:200]
        features_recon_sort[i] = features_recon_sort[i][:200]
        labels_sort[i] = labels_sort[i][:200]
    features_sort_ = np.vstack(features_sort)
    features_recon_sort_ = np.vstack(features_recon_sort)
    features_all_ = np.concatenate((features_sort_, features_recon_sort_),axis=0)
    labels_sort_ = np.hstack(labels_sort)
    print(features_sort_.shape, labels_sort_.shape, features_all_.shape)

    print('TSNEing')
    #feature_tsne=TSNE(n_components=2, init='pca', early_exaggeration=100, perplexity=100).fit_transform(features_sort_)
    feature_tsne_all=TSNE(n_components=2, init='pca').fit_transform(features_all_)
    print('TSNE Finished')

    feature_tsne = feature_tsne_all[:feature_tsne_all.shape[0]//2]
    feature_tsne_recon = feature_tsne_all[feature_tsne_all.shape[0]//2:]

    #plot
    color = plt.cm.rainbow(np.linspace(0, 1, 10))

    plt.figure(figsize=(9,7))
    for i in range(10):
        plt.scatter(feature_tsne[labels_sort_==i,0],feature_tsne[labels_sort_==i,1],marker='o',s=30,edgecolors='k',linewidths=1,c=color[i])
        plt.scatter(feature_tsne_recon[labels_sort_==i,0],feature_tsne_recon[labels_sort_==i,1],marker='s',s=30,edgecolors='k',linewidths=1,c=color[i])
    plt.xticks([])
    plt.yticks([])

    # fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    # im = ax.imshow(sim_mat, cmap='Blues')
    # fig.colorbar(im, pad=0.02, drawedges=0, ticks=[0, 0.5, 1])
    # ax.set_xticks(np.linspace(0, 20000, 6))
    # ax.set_yticks(np.linspace(0, 20000, 6))
    # [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()] 
    # [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
    # fig.tight_layout()

    save_dir = os.path.join(args.model_dir, 'figures', 'tsne')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"tsne_train_feature_epoch{epoch}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

def plot_random_gen(args, netG, ncomp=8, scale=0.5):
    '''
        ncomp: the number of components for random sampling
        scale: value to control sample range
    '''
    ### Generate random samples
    print('Load statistics from PCA results.')
    sig_vals_each_class = np.load(os.path.join(args.model_dir, f"figures/pca/sig_vals_each_class_epo{args.epoch}.npy"))
    components_each_class = np.load(os.path.join(args.model_dir, f"figures/pca/components_each_class_epo{args.epoch}.npy"))
    means_each_class = np.load(os.path.join(args.model_dir, f"figures/pca/means_each_class_epo{args.epoch}.npy"))
    
    random_images = []
    for i in range(10):
        sig_vals = np.array(sig_vals_each_class[i][:ncomp])
        var_vals = np.sqrt(sig_vals / np.sum(sig_vals))
        
        random_samples = np.random.normal(size=(64, ncomp))
        Z_random = means_each_class[i] + np.dot((scale * var_vals * random_samples), components_each_class[i][:ncomp]) # can modify scale to lower value to get more clear results  
        Z_random = Z_random / np.linalg.norm(Z_random, axis=1).reshape(-1,1)
        print(np.linalg.norm(Z_random, axis=1))
        X_recon_random = netG(torch.tensor(Z_random, dtype=torch.float).view(64,nz,1,1).to(device)).cpu().detach()
        print(X_recon_random.shape)
        random_images.append(X_recon_random)
        
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Random Recon Images")
        plt.imshow(np.transpose(vutils.make_grid(X_recon_random[:64], padding=2, normalize=True).cpu(), (1,2,0)))

        save_dir = os.path.join(args.model_dir, 'figures', 'images')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f"random_recon_images_epoch{args.epoch}_scale{scale}_class{i}.png")
        plt.savefig(file_name)
        print("Plot saved to: {}".format(file_name))

    random_images = torch.cat(([random_images[i][:8] for i in range(10)]), dim=0)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Random Recon Images")
    plt.imshow(np.transpose(vutils.make_grid(random_images, padding=2, normalize=True).cpu(), (1,2,0)))

    save_dir = os.path.join(args.model_dir, 'figures', 'images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"random_recon_images_epoch{args.epoch}_scale{scale}_ncomp{ncomp}_all_{manualSeed}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))


def plot_recon_imgs(train_X, train_X_bar, test_X, test_X_bar):
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Train Images")
    plt.imshow(np.transpose(vutils.make_grid(train_X[:64], padding=2, normalize=True).cpu(), (1,2,0)))
    
    save_dir = os.path.join(args.model_dir, 'figures', 'images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"train_images_epoch{args.epoch}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Recon Train Images")
    plt.imshow(np.transpose(vutils.make_grid(train_X_bar[:64], padding=2, normalize=True).cpu(), (1,2,0)))
    
    save_dir = os.path.join(args.model_dir, 'figures', 'images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"train_recon_images_epoch{args.epoch}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Test Images")
    plt.imshow(np.transpose(vutils.make_grid(test_X[:64], padding=2, normalize=True).cpu(), (1,2,0)))
    
    save_dir = os.path.join(args.model_dir, 'figures', 'images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"test_images_epoch{args.epoch}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Recon Test Images")
    plt.imshow(np.transpose(vutils.make_grid(test_X_bar[:64], padding=2, normalize=True).cpu(), (1,2,0)))
    
    save_dir = os.path.join(args.model_dir, 'figures', 'images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"test_recon_images_epoch{args.epoch}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))

def plot_linear_gen(args, netG, range_value=1, lin_sample_num=8):
    '''
        range_value: linspace value range
        lin_sample_num: linspace sample number
    '''
    ### Generate linspace samples
    sig_vals_each_class = np.load(os.path.join(args.model_dir, f"figures/pca/sig_vals_each_class_epo{args.epoch}.npy"))
    components_each_class = np.load(os.path.join(args.model_dir, f"figures/pca/components_each_class_epo{args.epoch}.npy"))
    means_each_class = np.load(os.path.join(args.model_dir, f"figures/pca/means_each_class_epo{args.epoch}.npy"))
    
    lin_gen_images = []
    for select_class in range(10):
        for j in range(4):
            sig_vals = np.array(sig_vals_each_class[select_class][j])
            var_vals = np.sqrt(sig_vals / np.sum(sig_vals))
            
            lin_samples = np.linspace(-range_value, range_value, lin_sample_num, endpoint=True)
            Z_lin = means_each_class[select_class] + np.dot(lin_samples.reshape(-1,1), components_each_class[select_class][j].reshape(1,-1)) # can modify 1 to lower value to get more clear results  
            Z_lin = Z_lin / np.linalg.norm(Z_lin, axis=1).reshape(-1,1)
            X_recon_lin = netG(torch.tensor(Z_lin, dtype=torch.float).view(lin_sample_num,nz,1,1).to(device)).cpu().detach()
            lin_gen_images.append(X_recon_lin)

            plt.figure(figsize=(8,8))
            plt.axis("off")
            plt.title("Lin Recon Images")
            plt.imshow(np.transpose(vutils.make_grid(X_recon_lin[:lin_sample_num], padding=2, normalize=True).cpu(), (1,2,0)))

            save_dir = os.path.join(args.model_dir, 'figures', 'images')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_name = os.path.join(save_dir, f"lin_recon_images_epoch{args.epoch}_class{select_class}_comp{j}_range{range_value}.png")
            plt.savefig(file_name)
            print("Plot saved to: {}".format(file_name))
    
    lin_gen_images = torch.cat(lin_gen_images,dim=0)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Lin Recon Images")
    plt.imshow(np.transpose(vutils.make_grid(lin_gen_images, nrow=16, padding=2, normalize=True).cpu(), (1,2,0)))

    save_dir = os.path.join(args.model_dir, 'figures', 'images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"lin_recon_images_all_epoch{args.epoch}_range{range_value}.png")
    plt.savefig(file_name)
    print("Plot saved to: {}".format(file_name))

def plot_nearest_component_class_X_hat(args, features, labels, epoch, train_X):
    """Find corresponding images to the nearests component per class. """
    features_sort, _ = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=10, stack=False)
    data_sort, _ = utils.sort_dataset(train_X, labels.numpy(), 
                            num_classes=10, stack=False)

    for class_ in range(10):
        nearest_data = []
        nearest_val = []
        pca = TruncatedSVD(n_components=10, random_state=10).fit(features_sort[class_])
        for j in range(8):
            proj = features_sort[class_] @ pca.components_.T[:, j]
            img_idx = np.argsort(np.abs(proj), axis=0)[::-1][:10]
            nearest_val.append(proj[img_idx])
            nearest_data.append(np.array(data_sort[class_])[img_idx])
        
        fig, ax = plt.subplots(ncols=10, nrows=8, figsize=(10, 10))
        for r in range(8):
            for c in range(10):
                ax[r, c].imshow(np.moveaxis(nearest_data[r][c],0,-1), cmap='gray')
                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])
                ax[r, c].spines['top'].set_visible(False)
                ax[r, c].spines['right'].set_visible(False)
                ax[r, c].spines['bottom'].set_linewidth(False)
                ax[r, c].spines['left'].set_linewidth(False)
                # ax[r, c].set_xlabel(f"proj: {nearest_val[r][c]:.2f}")
                if c == 0:
                    ax[r, c].set_ylabel(f"comp {r}")
        fig.tight_layout()

        ## save
        save_dir = os.path.join(args.model_dir, 'figures', 'nearcomp_class')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f"nearest_class{class_}_X_hat_{args.epoch}.png")
        fig.savefig(file_name)
        print("Plot saved to: {}".format(file_name))
        file_name = os.path.join(save_dir, f"nearest_class{class_}_X_hat_{args.epoch}.pdf")
        fig.savefig(file_name)
        print("Plot saved to: {}".format(file_name))
        plt.close()

def get_features_AE(encoder, decoder, trainloader, device, verbose=True):
    '''Extract all features out into one single batch. 
    
    Parameters:
        net (torch.nn.Module): get features using this model
        trainloader (torchvision.dataloader): dataloader for loading data
        verbose (bool): shows loading staus bar

    Returns:
        features (torch.tensor): with dimension (num_samples, feature_dimension)
        labels (torch.tensor): with dimension (num_samples, )
    '''
    X_all = []
    X_bar_all = []
    Z_all = []
    Z_bar_all = []
    labels_all = []
    if verbose:
        train_bar = tqdm(trainloader, desc="extracting all features from dataset")
    else:
        train_bar = trainloader
    for step, (X, labels) in enumerate(train_bar):
        Z = encoder(X.to(device))
        X_bar = decoder(Z.reshape(Z.shape[0],-1,1,1))
        Z_bar = encoder(X_bar.detach())

        X_all.append(X.cpu().detach())
        Z_all.append(Z.view(-1, Z.shape[1]).cpu().detach())
        X_bar_all.append(X_bar.cpu().detach())
        Z_bar_all.append(Z_bar.view(-1, Z_bar.shape[1]).cpu().detach())

        labels_all.append(labels)
        
    return torch.cat(X_all), torch.cat(Z_all), torch.cat(X_bar_all), torch.cat(Z_bar_all), torch.cat(labels_all)

class CostomAffineTransform:
    """Transform by one of the ways."""
    '''Choice:[angle, scale]'''

    def __init__(self, choices):
        self.choices = choices

    def __call__(self, x):    
        choice = random.choice(self.choices)
        x = FF.affine(x, angle=choice[0], scale=choice[1], translate=[0,0], shear=0)
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ploting')
    parser.add_argument('--model_dir', type=str, help='base directory for saving PyTorch model.')
    parser.add_argument('--loss', help='plot losses from training', action='store_true')
    parser.add_argument('--loss_log', help='plot losses from training', action='store_true')
    parser.add_argument('--hist', help='plot histogram of cosine similarity of features', action='store_true')
    parser.add_argument('--pca', help='plot PCA singular values of feautres', action='store_true')
    parser.add_argument('--tsne', help='plot tsne of feautres', action='store_true')
    parser.add_argument('--random_gen', help='plot random generated images', action='store_true')
    parser.add_argument('--lin_gen', help='plot linear generated images', action='store_true')
    parser.add_argument('--pca_epoch', help='plot PCA singular for different epochs', action='store_true')
    parser.add_argument('--nearcomp_sup', help='plot nearest component', action='store_true')
    parser.add_argument('--nearcomp_unsup', help='plot nearest component', action='store_true')
    parser.add_argument('--nearcomp_class', help='plot nearest component', action='store_true')
    parser.add_argument('--acc', help='plot accuracy over epochs', action='store_true')
    parser.add_argument('--traintest', help='plot train and test loss comparison plot', action='store_true')
    parser.add_argument('--heat', help='plot heatmap of cosine similarity between samples', action='store_true')
    parser.add_argument('--epoch', type=int, default=None, help='which epoch for evaluation')
    parser.add_argument('--n', type=int, default=1000, help='number of samples')
    parser.add_argument('--comp', type=int, default=30, help='number of components for PCA (default: 30)')
    parser.add_argument('--class_', type=int, default=None, help='which class for PCA (default: None)')
    parser.add_argument('--isize', type=int, default=32, help='input image size.')
    parser.add_argument('--nc', type=int, default=1, help='input image channels')
    parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## load data and model
    netG = Generator(ngpu=1, nz=args.nz, ngf=args.ngf, nc=args.nc).to(device)
    netD = Discriminator(ngpu=1, nz=args.nz, ndf=args.ndf, nc=args.nc).to(device) 
    D_ckpt_path = os.path.join(args.model_dir, 'checkpoints', 'model-D-epoch{}.pt'.format(args.epoch))
    print('Loading D checkpoint: {}'.format(D_ckpt_path))
    D_state_dict = torch.load(D_ckpt_path)
    netD.load_state_dict(D_state_dict)
    # netD.eval()

    G_ckpt_path = os.path.join(args.model_dir, 'checkpoints', 'model-G-epoch{}.pt'.format(args.epoch))
    print('Loading G checkpoint: {}'.format(G_ckpt_path))
    G_state_dict = torch.load(G_ckpt_path)
    netG.load_state_dict(G_state_dict)
    # netG.eval()

    # load dataset
    if args.dataset == 'CIFAR10':
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=False, num_workers=workers)
        testset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=workers)
    elif args.dataset == 'MNIST':
        transform = transforms.Compose(
                    [transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5)])
        trainset = datasets.MNIST(root='./data', train=True,
                                    download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=200,
                                                shuffle=False, num_workers=2)
        testset = datasets.MNIST(root='./data', train=False,
                                    download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=200,
                                                shuffle=False, num_workers=2)
    elif args.dataset == 'T_MNIST':
        transform = transforms.Compose(
                    [transforms.Resize(32),
                    transforms.ToTensor(),
                    CostomAffineTransform(choices=[[0, 1], [0, 1.5], [0, 0.5], [-45, 1], [45, 1]]),
                    transforms.Normalize(0.5, 0.5)])
        trainset = datasets.MNIST(root='./data', train=True,
                                    download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=200,
                                                shuffle=False, num_workers=2)
        testset = datasets.MNIST(root='./data', train=False,
                                    download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=200,
                                                shuffle=False, num_workers=2)

    train_X, train_Z, train_X_bar, train_Z_bar, train_labels = get_features_AE(netD, netG, trainloader, device)
    test_X, test_Z, test_X_bar, test_Z_bar, test_labels = get_features_AE(netD, netG, testloader, device)

    print(train_X.shape, train_Z.shape, train_X_bar.shape, train_Z_bar.shape, train_labels.shape)
    print(test_X.shape, test_Z.shape, test_X_bar.shape, test_Z_bar.shape, test_labels.shape)
    
    plot_recon_imgs(train_X, train_X_bar, test_X, test_X_bar) # Figure 3,5,8,10 -- draw reconstruction results

    if args.pca:
        plot_pca(args, train_Z, train_labels, args.epoch, args.class_)
    if args.nearcomp_sup:
        plot_nearest_component_supervised(args, train_Z, train_labels, args.epoch, trainset)
    if args.nearcomp_unsup:
        plot_nearest_component_unsupervised(args, train_Z, train_labels, args.epoch, trainset)
    if args.nearcomp_class:
        plot_nearest_component_class(args, train_Z, train_labels, args.epoch, trainset)
        plot_nearest_component_class_X_hat(args, train_Z, train_labels, args.epoch, train_X_bar) # Figure 6,11 -- draw reconstruction along components
    if args.hist:
        plot_hist(args, train_Z, train_labels, args.epoch)
    if args.heat:
        plot_heatmap_ZnZ_hat(args, train_Z, train_Z_bar, train_labels, args.epoch) # Figure 4 -- draw similarity heatmap of Z and Z_hat
    if args.tsne:
        plot_tsne_all(args, train_Z, train_Z_bar, train_labels, args.epoch)
        
    ## run --pca first
    if args.random_gen:
        plot_random_gen(args, netG) # Figure 9,12 -- draw random generated images
    if args.lin_gen:
        plot_linear_gen(args, netG)
    