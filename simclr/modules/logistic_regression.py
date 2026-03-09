import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes, bias=False):
        super(LogisticRegression, self).__init__()
        if not bias:
            print("Not using bias in the logistic regression model.")

        self.model = nn.Linear(n_features, n_classes, bias=bias)

    def forward(self, x):
        return self.model(x)
