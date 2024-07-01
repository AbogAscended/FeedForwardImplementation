import torch.nn as nn
import torch.nn.init as init


class FeedForward(nn.Module):
    def __init__(self, dropout_rate):
        super(FeedForward, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(287, 500),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(500, 500),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(500, 500),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(500, 500),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(500, 500),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(500, 500),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(500, 1)
        )

    def forward(self, x):
        return self.sequential(x)

    def init_weights(self):
        for layer in self.sequential:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
