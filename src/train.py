from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


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
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.nb_actions)
        ).to(self.device)

    def forward(self, x):
        return self.dqnet(x)

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

class ProjectAgent:
    
    def __init__(self, model_path = 'AgentDQN.pt'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.config = {'nb_actions' : 4,
                        'hidden_dim' : 256,
                        'nb_states' : 6}
        
        self.model_path = model_path

    def act(self, observation):
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()

    def save(self, path):
        pass

    def load(self):
        self.model = DQN(self.config).to(self.device)
        self.model.load_state_dict(
            torch.load(self.model_path, map_location = torch.device("cpu"))
        )
        self.model.eval()
