import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RouterQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(RouterQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.norm1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        return self.out(x)

class RouterAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-4, scaler=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.scaler = scaler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = RouterQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = RouterQNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state, epsilon=0.0):
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            q_values = self.q_net(state)
            return torch.argmax(q_values).item()

    def train(self, batch):
        states, actions, rewards, next_states = batch

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)

        q_values = self.q_net(states)
        q_val = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.q_net(next_states)
            next_actions = next_q.argmax(1)
            next_q_target = self.target_net(next_states)
            next_q_val = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        td_loss = F.mse_loss(q_val, rewards + self.gamma * next_q_val)

        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        return td_loss.item()
