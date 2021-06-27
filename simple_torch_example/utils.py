import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"


def top_n_accuracy(preds: np.ndarray,
                   targets: np.ndarray,
                   n_size: int) -> float:
    """
    Top N accuracy metric

    :param preds: array of unique sorted by confidence predicted labels
    :param targets: array of ground truth labels
    :param n_size:
    :return: Top N accuracy
    """
    return (targets.reshape(-1, 1) == preds[:, :n_size]).any(axis=1).mean()


def top_n_preds(labels: np.ndarray,
                dists: np.ndarray,
                n: int) -> np.ndarray:
    """
    Top N unique predicted labels

    :param labels: all predicted labels
    :param dists: distances array
    :param n: n predictions to consider
    :return: top n unique predicted labels
    """
    sorted_preds = labels[np.argsort(dists, axis=1)][:, 1:]
    # https://stackoverflow.com/questions/12926898/numpy-unique-without-sort
    top_n_preds = []
    for row in sorted_preds:
        indexes = np.unique(row, return_index=True)[1]
        preds = [row[index] for index in sorted(indexes)]
        top_n_preds.append(preds[:n])
    return np.array(top_n_preds)


@torch.no_grad()
def get_embeds(model: nn.Module, dataloader: DataLoader) -> np.ndarray:
    """
    Get images embeddings

    :param model: torch DL model
    :param dataloader: torch Dataloader
    :return: array of images embeddings and labels
    """
    embeddings = []
    labels = []
    model.eval()
    for batch in tqdm(dataloader):
        embedding = model.forward(batch["sample"].to(DEVICE))
        embeddings.append(embedding.cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        torch.cuda.empty_cache()

    return np.concatenate(embeddings), np.concatenate(labels)


def l2_dist(x: np.ndarray) -> np.ndarray:
    """
    L2 distance

    :param x: objects-features array of shape (#objects, #features)
    :return: square array of distances between objects
    """
    a = x.dot(x.T)
    b = np.diag(a)
    dist = np.sqrt(b.reshape(-1, 1) - 2 * a + b)
    return dist


def compute_accs(labels: np.ndarray,
                 dists: np.ndarray,
                 n: List[int] = [1, 5]) -> List[float]:
    top_preds = top_n_preds(labels, dists, max(n))
    accs = [top_n_accuracy(top_preds, labels, n_) for n_ in n]
    return accs


def set_seed(seed):
    """
    Set seed
    Source: # https://stackoverflow.com/questions/56354461/reproducibility-and-performance-in-pytorch
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)