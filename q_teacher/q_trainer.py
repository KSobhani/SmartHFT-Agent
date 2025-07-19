import numpy as np
from collections import deque
import random
from tqdm import trange

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

def train_with_q_teacher(agent, env, Q_star, num_steps=10000, epsilon=0.1, batch_size=64, target_update=100):
    replay_buffer = ReplayBuffer(capacity=10000)
    state = env.reset()
    rewards = []

    # مرحله 1: جمع‌آوری با پالیسی اکتشافی
    for _ in trange(num_steps // 2, desc="Collecting with ε-greedy"):
        action = agent.select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        current_pos_idx = env.action_space.index(env.position)
        q_star_vec = Q_star[env.current_step, current_pos_idx]

        replay_buffer.push((state, action, reward, next_state, q_star_vec))
        state = next_state if not done else env.reset()

    # مرحله 2: جمع‌آوری با actor بهینه (argmax Q*)
    for _ in trange(num_steps // 2, desc="Collecting with optimal actor"):
        current_pos_idx = env.action_space.index(env.position)
        q_star_vec = Q_star[env.current_step - 1, current_pos_idx]  # shape = [action_dim]

        current_pos_idx = env.action_space.index(env.position)
        optimal_action = np.argmax(q_star_vec[current_pos_idx])
        next_state, reward, done, _ = env.step(optimal_action)
        replay_buffer.push((state, optimal_action, reward, next_state, q_star_vec))
        state = next_state if not done else env.reset()

    # مرحله 3: آموزش با نمونه‌گیری مینی‌بچ
    for step in trange(num_steps, desc="Training DDQN agent"):
        if len(replay_buffer) < batch_size:
            continue

        batch = replay_buffer.sample(batch_size)
        td_loss, kl_loss, total_loss = agent.train(batch)

        if step % target_update == 0:
            agent.update_target()

    return agent
