import torch.nn as nn
import torchvision

from simclr.modules.resnet_hacks import modify_resnet_model
from simclr.modules.identity import Identity


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features, n_features_latent=-1):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features
        self.n_features_latent = n_features_latent

        # Replace the fc layer with an Identity function
        if hasattr(encoder, "fc"):
            self.encoder.fc = Identity()
        elif hasattr(encoder, "classifier"):
            self.encoder.classifier = Identity()
        elif hasattr(encoder, "linear"):
            self.encoder.linear = Identity()
        else:
            raise RuntimeError("Unknown classifier structure specified.")

        if self.n_features_latent != -1:
            print("Compressing features to latent space.{}".format(n_features_latent))
            self.linear_compress = nn.Sequential(
                nn.Linear(n_features, n_features_latent),
                nn.BatchNorm1d(n_features_latent),
                nn.ReLU(),
            )
            # self.linear_compress = nn.Conv1d(n_features, n_features_latent, 1)
            self.n_features = n_features_latent

        # breakpoint()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        if self.n_features_latent != -1:
            # h_i = h_i.unsqueeze(-1)
            # h_j = h_j.unsqueeze(-1)

            h_i = self.linear_compress(h_i)
            h_j = self.linear_compress(h_j)

            # h_i = h_i.squeeze(-1)
            # h_j = h_j.squeeze(-1)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j
