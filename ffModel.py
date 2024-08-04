import torch.nn as nn
import torch.nn.init as init


# Actual FeedForward class which needs to override the forward method for a forward pass from module parent class
class FeedForward(nn.Module):
    def __init__(self, dropout_rate):
        super(FeedForward, self).__init__()

        # Using a sequential container to sequentially pass the input through each layer
        # One input layer of 287 which is the size of the input. 5 Hidden layers with 500 units each
        # Since the data set is uni-variate regression the output is of size 1.
        # Im using ELU activation because this model has a hard time with exploding gradients and i found
        # ELU was the best activation function that i tested which minimized the amount of NANs.
        # Drop out rate is a variable so that optuna can try to find the best value.
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

    # forward simply passes inputs through sequential container
    def forward(self, x):
        return self.sequential(x)

    # this initializes the weights with HE initialization aka kaiming_uniform in pytorch.
    def init_weights(self):
        for layer in self.sequential:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
