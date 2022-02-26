from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from torch.utils.data import DataLoader
import utils
import torchvision.utils as vutils
import os
import argparse
from torch_mimicry.nets import sngan

import torch
from mcrgan.models import customSNGANDiscriminator128
from mcrgan.datasets import celeba_dataset
from utils.utils import extract_features, sort_dataset
import random
torch.multiprocessing.set_sharing_strategy('file_system')

manualSeed = 2
random.seed(manualSeed)  # python random seed
torch.manual_seed(manualSeed)  # pytorch random seed
np.random.seed(manualSeed)  # numpy random seed
torch.backends.cudnn.deterministic = True


def plot_pca(features, labels, model_dir, ncomp, epoch='end', select_label=None):
    """Plot PCA of learned features. """
    ## create save folder
    pca_dir = os.path.join(model_dir, 'figures', 'pca')
    if not os.path.exists(pca_dir):
        os.makedirs(pca_dir)

    ## perform PCA on features
    n_comp = np.min([ncomp, features.shape[1]])
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
            color_c = 'green' if c == select_label else color[c]
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


def plot_linear_gen_on_images(train_Z, netG, model_dir, epoch='end'):
    # Generate linspace samples
    components_each_class = np.load(
        os.path.join(model_dir, f"figures/pca/components_each_class_epo{epoch}.npy"))
    means_each_class = np.load(os.path.join(model_dir, f"figures/pca/means_each_class_epo{epoch}.npy"))
    print('mean norm:', np.linalg.norm(means_each_class, axis=1))

    range_value = 1
    lin_sample_num = 6
    for j in range(len(components_each_class[0])):
        lin_gen_images = []
        for select_image in range(10):
            lin_samples = np.linspace(0, range_value, lin_sample_num, endpoint=True)
            Z_lin = train_Z[select_image] + \
                    np.dot(lin_samples.reshape(-1, 1), components_each_class[0][j].reshape(1, -1))  # can modify 1 to lower value to get more clear results
            # normalization
            Z_lin = Z_lin / np.linalg.norm(Z_lin, axis=1).reshape(-1, 1)  # normalization
            print(np.linalg.norm(Z_lin, axis=1))

            X_recon_lin = netG(
                torch.tensor(Z_lin, dtype=torch.float).view(lin_sample_num, 128).cuda()).cpu().detach()
            lin_gen_images.append(X_recon_lin)

        lin_gen_images = torch.cat(lin_gen_images, dim=0)
        plt.figure(figsize=(40, 40))
        plt.axis("off")
        # plt.title("Lin Recon Images")
        plt.imshow(np.transpose(vutils.make_grid(lin_gen_images, nrow=lin_sample_num, padding=2, normalize=True).cpu(),
                                (1, 2, 0)))

        save_dir = os.path.join(model_dir, 'figures', 'images')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f"lin_select_recon_images_epoch{epoch}_comp{j}_range{range_value}.png")
        plt.savefig(file_name)
        print("Plot saved to: {}".format(file_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ploting')
    parser.add_argument('--model_dir', default='./logs/visualization/celebA_1', type=str, required=True,
                        help='base directory for saving PyTorch model.')
    parser.add_argument('--checkpoint_dir', default='./logs/visualization/celebA_1', type=str, required=True,
                        help='directory to checkpoint.')
    parser.add_argument('--data_root', default='./data2/celeba/img_align_celeba', type=str, required=True,
                        help='directory to dataset.')
    parser.add_argument('--comp', type=int, default=127, help='number of components for PCA (default: 30)')
    args = parser.parse_args()

    # load model
    netG = sngan.SNGANGenerator128().cuda()
    netD = customSNGANDiscriminator128().cuda()

    state_dict_G = torch.load(args.checkpoint_dir + 'netG/netG_100000_steps.pth')['model_state_dict']
    state_dict_D = torch.load(args.checkpoint_dir + 'netD/netD_100000_steps.pth')['model_state_dict']
    netG.load_state_dict(state_dict_G)
    netD.load_state_dict(state_dict_D)

    netG.eval()
    netD.eval()

    # load dataset
    dataset = celeba_dataset(root=args.data_root, size=128)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=16)

    # extracting features
    _, train_Z, _, train_Z_bar, train_labels = extract_features(dataloader, netD, netG)

    # viz generation along different pca directions
    plot_pca(train_Z, train_labels, args.model_dir, args.n_comp)
    plot_linear_gen_on_images(train_Z, netG, args.model_dir)
