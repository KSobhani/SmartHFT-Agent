import numpy as np
from collections import defaultdict


def evaluate_agent(agent, env, Q_star):
    # [INFO] شروع ارزیابی agent
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state, epsilon=0.0)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    return total_reward

def select_best_agents(df, trained_agents, action_values, label_col='trend_label', initial_positions=None, return_stats=False):
    if initial_positions is None:
        initial_positions = [0]

    segments = defaultdict(list)
    for label in np.unique(df[label_col]):
        indices = df[df[label_col] == label].index
        starts = indices[::60]
        for s in starts:
            if s + 62 < len(df):
                possible_columns = [
                    'price_open', 'price_high', 'price_low', 'price_close',
                    'taker_volume_buy', 'taker_volume_sell',
                    'volume_diff', 'spread',
                    'liquid_vol_buy', 'liquid_vol_sell',
                    'open_interest', 'delta_lob', 'corr_lob_feat',
                    'mid_price', 'imbalance', 'liquidity_pressure'
                ]
                selected_columns = [col for col in possible_columns if col in df.columns]
                segment = df[selected_columns].iloc[s:s+62].reset_index(drop=True)
                segments[label].append(segment)

    selected = dict()
    selection_stats = dict()
    all_agent_rewards = dict()

    from tqdm import tqdm as tqdm_global
    for label in tqdm_global(segments, desc="Processing labels"):
        for pos in tqdm_global(initial_positions, desc=f"label={label}", leave=False):
            best_reward = -np.inf
            best_agent = None
            best_beta = None
            rewards_per_beta = dict()

            

            print(f"[INFO] شروع ارزیابی label={label} - pos={pos}")
            for beta, agent in tqdm_global(trained_agents, desc=f"Evaluating βs", leave=False):
                rewards = []
                for seg in segments[label]:
                    from envs.market_env import MarketEnv
                    from q_teacher.optimal_value import compute_optimal_q
                    env = MarketEnv(seg, action_values)
                    env.position = pos
                    env.current_step = 0
                    Q_star = compute_optimal_q(seg, action_values)
                    reward = evaluate_agent(agent, env, Q_star)
                    rewards.append(reward)
                avg_reward = np.mean(rewards)
                print(f"    → β={beta} → avg_reward={avg_reward:.2f}")
                rewards_per_beta[beta] = avg_reward
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_agent = agent
                    best_beta = beta

            selected[(label, pos)] = best_agent
            selection_stats[(label, pos)] = {
                "beta": best_beta,
                "avg_reward": best_reward
            }
            all_agent_rewards[(label, pos)] = rewards_per_beta

    if return_stats:
        return selected, selection_stats, all_agent_rewards
    else:
        return selected
