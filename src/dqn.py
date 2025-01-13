import torch
import torch.nn as nn


class DQN (torch.nn.Module):
    def __init__(self, config):
        super(DQN, self).__init__()
        
        # Setup the parameters of the problem
        self.nb_actions = config['nb_actions']
        self.nb_states = config['nb_states']
        self.hidden_dim = config['hidden_dim']

        # Setup the device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the DQN block
        self.InitNNBlock()

    def InitNNBlock(self):
        # Initialize the DQN
        self.dqnet = nn.Sequential(
            nn.Linear(self.nb_states, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.nb_actions)
        ).to(self.device)

    def forward(self, x):
        return self.dqnet(x)