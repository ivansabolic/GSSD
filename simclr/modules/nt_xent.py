import torch
import torch.nn as nn
import torch.distributed as dist
from .gather import GatherLayer


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature # (2N, 2N)

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size) # (N,) sličnost među pozitiivnim parovima
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size) # (N,) sličnost među pozitiivnim parovima

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1) # N, 2N-2 za svaki primjerak, sličnost sa svim ostalim primjerima

        labels = torch.zeros(N).to(positive_samples.device).long() # N, 0 jer se traži cross entropy za ovaj prvi primjerak iz donjeg cata
        logits = torch.cat((positive_samples, negative_samples), dim=1) # N, 2N-1
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class BackdoorNT_Xent(NT_Xent):
    def forward(self, z_i, labels):
        # N = 2 * self.batch_size * self.world_size

        # z = torch.cat((z_i, z_j), dim=0)
        z = z_i
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature  # (2N, 2N)

        # similarity to different labels
        mask = labels.unsqueeze(1) != labels.unsqueeze(0).to(z.device)
        sim_to_diffent_classes = sim.masked_fill(mask, float('-inf'))

        # sum for each sample
        sim_num = sim_to_diffent_classes.logsumexp(dim=1)

        # similarity to all besides itself
        mask = torch.eye(sim.shape[0]).bool().to(z.device)
        sim_to_all = sim.masked_fill(mask, float('-inf'))

        # sum for each sample
        sim_den = sim_to_all.logsumexp(dim=1)

        loss = - (sim_num - sim_den)

        return loss.mean()


