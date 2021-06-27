import torch.nn as nn
import torch
from ..utils import l2_dist, compute_accs


class SomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
        )
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, triplet):
        anchor = self.net(triplet[0])
        positive = self.net(triplet[1])
        negative = self.net(triplet[2])
        return anchor, positive, negative

    def compute_all(self, triplet, labels):
        # computes batch-wise loss and accuracy
        loss = self.triplet_loss(*triplet, reduction="none")

        all_embeds = torch.cat(triplet).cpu().detach().numpy()
        all_labels = torch.cat(labels).cpu().detach().numpy()

        dists = l2_dist(all_embeds)
        accs = compute_accs(all_labels, dists)

        return loss, dict(acc_1=accs[0], acc_5=accs[1])