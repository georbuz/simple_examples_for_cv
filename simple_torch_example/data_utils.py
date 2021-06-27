from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset, Sampler
import torch
from typing import Union, Tuple, List


class CelebADataset(Dataset):
    def __init__(self, train: bool = True) -> None:
        self._path = "celeba_data/"
        self.images_dir_path = self._path + "images/"
        self.file = self._path + "train.csv" if train else self._path + "val.csv"
        self.header = pd.read_csv(self.file)

    def __len__(self) -> int:
        return len(self.header)

    def __getitem__(self, index: int) -> dict:
        img_name, label = self.header.iloc[index, :]

        img_path = Path(self.images_dir_path, img_name)
        img = self._read_img(img_path)

        return dict(sample=img, label=label)

    @staticmethod
    def _read_img(img_path: Path) -> np.ndarray:
        img = cv2.imread(str(img_path.resolve()))
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1)) / 255.

        return img


class SimpleDataset(Dataset):
    """ Triplet dataset """
    def __init__(self, embeddings: torch.tensor,
                 labels: torch.tensor) -> None:

        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: Union[int, Tuple[int], List[int]]) -> dict:
        """ If idx is a single int - return one sample, otherwise return triplet"""
        if isinstance(idx, int):
            sample = self.embeddings[idx]
            label = self.labels[idx]

            return dict(
                sample=sample,
                label=label
            )
        else:
            assert len(idx) == 3

            anc_idx, pos_idx, neg_idx = idx

            samples = self.embeddings[anc_idx], self.embeddings[pos_idx], self.embeddings[neg_idx]
            labels = self.labels[anc_idx], self.labels[pos_idx], self.labels[neg_idx]

            return dict(
                samples=samples,
                labels=labels
            )


class SimpleTripletSampler(Sampler):
    """ Triplet sampler """
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for anchor_idx in range(len(self.dataset)):
            positive_idx = self._mine_positive(anchor_idx)
            negative_idx = self._mine_negative(anchor_idx)
            # if anchor has no positive samples -> dont yield triplet
            if positive_idx == -1:
                continue
            yield anchor_idx, positive_idx, negative_idx

    def _mine_positive(self, anchor_idx: int):

        anchor_label = self.dataset.labels[anchor_idx]
        pos_idxs = torch.nonzero(self.dataset.labels == anchor_label)
        pos_idx = pos_idxs[np.random.randint(low=0, high=pos_idxs.shape[0])]

        # no positive samples for anchor
        if len(pos_idxs) == 1:
            return -1

        return pos_idx.squeeze()

    def _mine_negative(self, anchor_idx: int):

        anchor_label = self.dataset.labels[anchor_idx]
        neg_idxs = torch.nonzero(self.dataset.labels != anchor_label)
        neg_idx = neg_idxs[np.random.randint(low=0, high=neg_idxs.shape[0])]

        return neg_idx.squeeze()