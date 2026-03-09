import torch
from torch import nn
import numpy as np
class GenerativeClassifier(nn.Module):
    def __init__(self, num_classes, latent_dim, init_latent_scale=5):
        self.num_classes = num_classes
        self.inn = ...

        mu_populate_dims = latent_dim
        self.mu = nn.Parameter(torch.zeros(1, num_classes, latent_dim))

        init_scale = init_latent_scale / np.sqrt(2 * mu_populate_dims // self.n_classes)
        for k in range(mu_populate_dims // self.n_classes):
            self.mu.data[0, :, self.n_classes * k: self.n_classes * (k + 1)] = init_scale * torch.eye(self.num_classes)

        self.phi = nn.Parameter(torch.zeros(self.num_classes))
        self.trainable_params = list(self.inn.parameters())
        self.trainable_params = list(filter(lambda p: p.requires_grad, self.trainable_params))
        self.trainable_params += [self.mu, self.phi]

        self.train_mu = True
        self.train_phi = False
        self.train_inn = True

        optimizer = ...

        base_lr = ...

        optimizer_params = [ {'params':list(filter(lambda p: p.requires_grad, self.inn.parameters()))},]

        if self.train_mu:
            optimizer_params.append({'params':[self.mu], 'lr':base_lr})

            if optimizer == 'SGD':
                optimizer_params[-1]['momentum'] = 0.9
            if optimizer == 'Adam':
                optimizer_params[-1]['betas'] = (0.9, 0.999)

        if self.train_phi:
            pass

        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(optimizer_params, base_lr,
                                             momentum=0.9,
                                             weight_decay=1e-4,
                                             )
        elif optimizer == 'ADAM':
            self.optimizer = torch.optim.Adam(optimizer_params, base_lr,
                                              betas=(0.9, 0.999),
                                              weight_decay=1e-4,
                                              )

        def cluster_distances(self, z, y=None):
            if y is not None:
                mu = torch.mm(z.t().detach(), y.round())
                mu = mu / torch.sum(y, dim=0, keepdim=True)
                mu = mu.t().view(1, self.n_classes, -1)
                mu = 0.005 * mu + 0.995 * self.mu.data
                self.mu.data = mu.data

            z_i_z_i = torch.sum(z ** 2, dim=1, keepdim=True)  # batchsize x n_classes
            mu_j_mu_j = torch.sum(self.mu ** 2, dim=2)  # 1 x n_classes
            z_i_mu_j = torch.mm(z, self.mu.squeeze().t())  # batchsize x n_classes

            return -2 * z_i_mu_j + z_i_z_i + mu_j_mu_j

        def mu_pairwise_dist(self):

            mu_i_mu_j = self.mu.squeeze().mm(self.mu.squeeze().t())
            mu_i_mu_i = torch.sum(self.mu.squeeze() ** 2, 1, keepdim=True).expand(self.n_classes, self.n_classes)

            dist = mu_i_mu_i + mu_i_mu_i.t() - 2 * mu_i_mu_j
            return torch.masked_select(dist, (1 - torch.eye(self.n_classes).cuda()).bool()).clamp(min=0.)


        def forward(self, x, y=None):
            z, log_jac_det = self.inn(x, y)

            if y is None:
                return z, log_jac_det

            # Compute cluster distances
            dist = self.cluster_distances(z, y)

            # Compute log p(y)
            log_py = -0.5 * dist + self.phi
            log_py = log_py - torch.logsumexp(log_py, dim=1, keepdim=True)

            # Compute log p(x)
            log_px = log_py - log_jac_det

            return z, log_px, log_py