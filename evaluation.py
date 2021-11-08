from scipy.stats import entropy
import torch_mimicry.metrics.fid as fid
from utils.inception import InceptionV3
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Union, Tuple
from utils.inceptionII import InceptionV3 as IC
from tqdm import tqdm
import argparse
from mcrgan.datasets import get_dataloader
from mcrgan.models import get_models
from mcrgan.default import _C as config
from mcrgan.default import update_config


def get_inception_feature(
    images: Union[List[torch.FloatTensor], DataLoader],
    dims: List[int],
    batch_size: int = 50,
    use_torch: bool = False,
    verbose: bool = False,
    device: torch.device = torch.device('cuda:0'),
) -> Union[torch.FloatTensor, np.ndarray]:
    """Calculate Inception Score and FID.
    For each image, only a forward propagation is required to
    calculating features for FID and Inception Score.

    Args:
        images: List of tensor or torch.utils.data.Dataloader. The return image
            must be float tensor of range [0, 1].
        dims: List of int, see InceptionV3.BLOCK_INDEX_BY_DIM for
            available dimension.
        batch_size: int, The batch size for calculating activations. If
            `images` is torch.utils.data.Dataloader, this argument is
            ignored.
        use_torch: bool. The default value is False and the backend is same as
            official implementation, i.e., numpy. If use_torch is enableb,
            the backend linalg is implemented by torch, the results are not
            guaranteed to be consistent with numpy, but the speed can be
            accelerated by GPU.
        verbose: Set verbose to False for disabling progress bar. Otherwise,
            the progress bar is showing when calculating activations.
        device: the torch device which is used to calculate inception feature
    Returns:
        inception_score: float tuple, (mean, std)
        fid: float
    """
    assert all(dim in IC.BLOCK_INDEX_BY_DIM for dim in dims)

    is_dataloader = isinstance(images, DataLoader)
    if is_dataloader:
        num_images = min(len(images.dataset), images.batch_size * len(images))
        batch_size = images.batch_size
    else:
        num_images = len(images)

    block_idxs = [IC.BLOCK_INDEX_BY_DIM[dim] for dim in dims]
    model = IC(block_idxs).to(device)
    model.eval()

    if use_torch:
        features = [torch.empty((num_images, dim)).to(device) for dim in dims]
    else:
        features = [np.empty((num_images, dim)) for dim in dims]

    pbar = tqdm(
        total=num_images, dynamic_ncols=True, leave=False,
        disable=not verbose, desc="get_inception_feature")
    looper = iter(images)
    start = 0
    while start < num_images:
        # get a batch of images from iterator
        if is_dataloader:
            batch_images = next(looper)
        else:
            batch_images = images[start: start + batch_size]
        end = start + len(batch_images)

        # calculate inception feature
        batch_images = torch.tensor(batch_images).to(device)
        with torch.no_grad():
            outputs = model(batch_images)
            for feature, output, dim in zip(features, outputs, dims):
                if use_torch:
                    feature[start: end] = output.view(-1, dim)
                else:
                    feature[start: end] = output.view(-1, dim).cpu().numpy()
        start = end
        pbar.update(len(batch_images))
    pbar.close()
    return features

# Change batch_size here with the capability of your machine
def cal_X(netG, netD, X, batch_size=1000):
    num_samples = len(X)
    num_batches = num_samples // batch_size

    # Get images
    images = []
    with torch.no_grad():
        start = 0
        end = min(num_samples, start+batch_size)
        for idx in range(num_batches):
            X_batch = X[start:end]
            Z = netD(X_batch.detach())
            fake_images = netG(torch.reshape(Z, (len(Z), 128))).detach().cpu()
            images.append(fake_images)
            start = end
            end = min(num_samples, start+batch_size)

    images = torch.cat(images, 0)

    #images = (images.cpu().numpy()+1.0)/2
    #X = (X.cpu().numpy()+1.0)/2

    images = images.cpu().numpy()
    X = X.cpu().numpy()
    #images = _normalize_images(images)
    #X = _normalize_images(X)

    return X, images

def __calc_is(preds, n_split, return_each_score=False):
    """
    regularly, return (is_mean, is_std)
    if n_split==1 and return_each_score==True:
        return (scores, 0)
        # scores is a list with len(scores) = n_img = preds.shape[0]
    """

    n_img = preds.shape[0]
    # Now compute the mean kl-div
    split_scores = []
    for k in range(n_split):
        part = preds[k * (n_img // n_split): (k + 1) * (n_img // n_split), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
        if n_split == 1 and return_each_score:
            return scores, 0
    return np.mean(split_scores), np.std(split_scores)


def calculate_inception_score(
    probs: Union[torch.FloatTensor, np.ndarray],
    splits: int = 10,
    use_torch: bool = False,
) -> Tuple[float, float]:
    # Inception Score
    scores = []
    for i in range(splits):
        part = probs[
            (i * probs.shape[0] // splits):
            ((i + 1) * probs.shape[0] // splits), :]
        if use_torch:
            kl = part * (
                torch.log(part) -
                torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        else:
            kl = part * (
                np.log(part) -
                np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
    if use_torch:
        scores = torch.stack(scores)
        inception_score = torch.mean(scores).cpu().item()
        std = torch.std(scores).cpu().item()
    else:
        inception_score, std = (np.mean(scores), np.std(scores))
    del probs, scores
    return inception_score, std



def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='should add the .yaml file',
                        required=True,
                        type=str,
                        )
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)
    #
    # return args


def main():

    # -----------------------
    parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # The Batch Size here is the amount of data you want to test your FID
    trainloader, dataset = get_dataloader(
        data_name=config.DATA.DATASET,
        root=config.DATA.ROOT,
        batch_size=config.EVAL.DATA_SAMPLE,
        num_workers=config.CUDNN.WORKERS
    )

    # Define models and optimizers
    netD, netG = get_models(config.DATA.DATASET, device)

    netD_state_dict, netG_state_dict = torch.load(config.EVAL.NETD_CKPT), torch.load(config.EVAL.NETG_CKPT)

    netG.module.load_state_dict(netG_state_dict["model_state_dict"])
    netD.module.load_state_dict(netD_state_dict["model_state_dict"])
    netG.cuda()
    netD.cuda()

    inception_model = InceptionV3()

    for i, (data, label) in enumerate(trainloader):

        real_images, fake_images = cal_X(netG, netD, data)
        acts, probs = get_inception_feature(
            (fake_images + 1)/2., dims=[2048, 1008], use_torch=True)
        inception_score, std = calculate_inception_score(probs, 10, True)
        print("Inception Score:", inception_score, "Inception STD:", std)

        # Can change device here
        mu_fake, sigma_fake = fid.calculate_activation_statistics(
            fake_images, inception_model, 50
        )
        mu_real, sigma_real = fid.calculate_activation_statistics(
            real_images, inception_model, 50
        )
        fid_score = fid.calculate_frechet_distance(
            mu_fake, sigma_fake, mu_real, sigma_real
        )
        print("FID is:", fid_score)
        break


if __name__ == '__main__':
    main()
