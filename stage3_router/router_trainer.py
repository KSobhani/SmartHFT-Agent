# stage3_router/router_trainer.py

import numpy as np
import random
from collections import deque
from .router_agent import RouterAgent

def train_router(env, router=None, num_episodes=200,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 batch_size=32, buffer_size=10000, target_update=10):
    """
    اگر router از بیرون پاس داده شده باشد، از آن استفاده می‌کند،
    در غیر این صورت یک RouterAgent جدید می‌سازد.
    """
    # ساخت یا استفاده از RouterAgent
    if router is None:
        state_dim = env._get_state().shape[0]
        action_dim = len(env.action_space)
        router = RouterAgent(state_dim, action_dim)

    replay_buffer = deque(maxlen=buffer_size)
    epsilon = epsilon_start
    rewards_all = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = router.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.append((state, action, reward, next_state))
            state = next_state
            episode_reward += reward

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                loss = router.train(list(zip(*batch)))

        # به‌روزرسانی ε و هدف
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_all.append(episode_reward)
        if episode % target_update == 0:
            router.update_target()
            print(f"Episode {episode} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.4f}")

    return router, rewards_all
