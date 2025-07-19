import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class DDQNAgent:
    def __init__(self, state_dim, action_dim, alpha=1.0, gamma=0.99, lr=1e-4,beta=None):
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def train(self, batch):
        states, actions, rewards, next_states, q_star = batch

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        q_star = torch.FloatTensor(q_star).to(self.device)

        q_values = self.q_net(states)
        q_val = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.q_net(next_states)
            next_actions = next_q.argmax(1)
            next_q_target = self.target_net(next_states)
            next_q_val = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        td_loss = F.mse_loss(q_val, rewards + self.gamma * next_q_val)

        # KL Divergence term between agent Q and Q* from Q-Teacher
        log_probs = F.log_softmax(q_values, dim=1)
        q_star_probs = F.softmax(q_star, dim=1)
        if q_star_probs.ndim == 1:
            q_star_probs = q_star_probs.unsqueeze(0).expand_as(log_probs)

        kl_loss = F.kl_div(log_probs, q_star_probs, reduction='batchmean')

        loss = td_loss + self.alpha * kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return td_loss.item(), kl_loss.item(), loss.item()
