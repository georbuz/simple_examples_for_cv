import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from ..data_utils import SimpleTripletSampler
from tqdm import tqdm
from collections import defaultdict


class Trainer:
    """ Trainer class """
    def __init__(self, model: nn.Module,
                 optimizer,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 tboard_log_dir: str = "./tboard_logs/",
                 batch_size: int = 128,
                 n_hardest: int = 256):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.train_sampler = SimpleTripletSampler(train_dataset)
        self.val_sampler = SimpleTripletSampler(val_dataset)
        self.hard_selector = nn.TripletMarginLoss(margin=1.0, p=2, reduction='none')
        self.n_hardest = n_hardest
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)

        self.global_step = 0
        self.train_writer = SummaryWriter(log_dir=tboard_log_dir + "train/")
        self.val_writer = SummaryWriter(log_dir=tboard_log_dir + "val/")

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    @torch.no_grad()
    def select_hardest(self, sample_losses: torch.tensor) -> np.ndarray:
        """ Hard miner """
        hard_idxs = list(sample_losses.topk(self.n_hardest).indices.detach().cpu().numpy())
        return hard_idxs

    def train(self, num_epochs: int):
        """ Train loop """
        model = self.model
        optimizer = self.optimizer

        train_loader = DataLoader(self.train_dataset,
                                  sampler=self.train_sampler,
                                  batch_size=self.batch_size)
        val_loader = DataLoader(self.val_dataset,
                                sampler=self.val_sampler,
                                batch_size=self.batch_size)
        best_loss = float('inf')

        for epoch in range(num_epochs):
            model.train()
            train_losses = []
            for batch in tqdm(train_loader):
                batch = {k: [v_.to(self.device) for v_ in v] for k, v in batch.items()}
                samples, labels = batch["samples"], batch["labels"]
                samples = model.forward(samples)

                # triplet loss for each sample
                sample_losses, _ = model.compute_all(samples, labels)

                hardest_idxs = self.select_hardest(sample_losses)
                labels = labels[0][hardest_idxs], labels[1][hardest_idxs], labels[2][hardest_idxs]
                samples = samples[0][hardest_idxs], samples[1][hardest_idxs], samples[2][hardest_idxs]

                loss, details = model.compute_all(samples, labels)
                loss = loss.mean()
                train_losses.append(loss.item())

                for k, v in details.items():
                    self.train_writer.add_scalar(k, v, global_step=self.global_step)
                self.global_step += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mean_train_loss = np.mean(train_losses)

            model.eval()
            val_losses = []
            val_logs = defaultdict(list)
            for batch in tqdm(val_loader):
                batch = {k: [v_.to(self.device) for v_ in v] for k, v in batch.items()}

                samples, labels = batch["samples"], batch["labels"]
                samples = model.forward(samples)

                loss, details = model.compute_all(samples, labels)
                val_losses.append(loss.item())
                for k, v in details.items():
                    val_logs[k].append(v)

            val_logs = {k: np.mean(v) for k, v in val_logs.items()}
            for k, v in val_logs.items():
                self.val_writer.add_scalar(k, v, global_step=self.global_step)

            val_loss = np.mean(val_losses)

            if val_loss < best_loss:
                self.save_checkpoint("./best_checkpoint.pth")
                best_loss = val_loss

            print("Batch mean train Loss:  %.4f\tBatch mean val Loss:  %.4f" % (mean_train_loss, val_loss))
            self.scheduler.step(val_loss)
