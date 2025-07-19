import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from envs.market_env import MarketEnv
from agents.ddqn_agent import DDQNAgent
from q_teacher.q_trainer import train_with_q_teacher
from q_teacher.optimal_value import compute_optimal_q
from stage2_diverse_pool.sampler import compute_return_rates, biased_sampling_indices


def train_multiple_agents(df, action_values, beta_list, window=60, num_agents_per_beta=5):
    """
    آموزش چندین عامل با biasهای مختلف نسبت به روند بازار، مطابق مرحله دوم مقاله EarnHFT.
    فقط روی سگمنت‌های دارای trend_label انجام می‌شود.
    """
    trained_agents = []
    segments_by_label = defaultdict(list)

    print("[INFO] استخراج سگمنت‌ها بر اساس trend_label ...")
    for label in np.unique(df['trend_label']):
        indices = df[df['trend_label'] == label].index
        for start in range(indices[0], indices[-1] - window - 60, window):
            segment = df.iloc[start:start + window + 60].reset_index(drop=True)
            if len(segment) == window + 60:
                segments_by_label[label].append(segment)

    print("[INFO] مجموع سگمنت‌ها:", sum(len(v) for v in segments_by_label.values()))

    for beta in tqdm(beta_list, desc="β values"):
        all_returns = []
        segment_map = []

        for label, seglist in segments_by_label.items():
            for seg in seglist:
                r = (seg['price_close'].iloc[window] - seg['price_close'].iloc[0]) / seg['price_close'].iloc[0]
                all_returns.append(r)
                segment_map.append((label, seg))

        all_returns = np.array(all_returns)
        print(f"[DEBUG] β={beta} → مجموع سگمنت‌ها = {len(all_returns)}")

        try:
            probs = biased_sampling_indices(all_returns, beta)
        except Exception as e:
            print(f"[ERROR] محاسبه probs برای β={beta} شکست خورد: {e}")
            continue

        for i in tqdm(range(num_agents_per_beta), desc=f"Agents for β={beta}", leave=False):
            try:
                idx = np.random.choice(len(all_returns), p=probs)
                label, segment = segment_map[idx]
                print(f"→ شروع آموزش agent {i+1}/{num_agents_per_beta} برای β={beta}, trend_label={label}")

                env = MarketEnv(segment, action_values)
                Q_star = compute_optimal_q(segment, action_values)
                sample_state = env.reset()
                state_dim = sample_state.shape[0]
                action_dim = len(action_values)
                agent = DDQNAgent(state_dim, action_dim, alpha=1.0)
                trained_agent = train_with_q_teacher(agent, env, Q_star, num_steps=2000, epsilon=0.1)
                trained_agents.append((beta, trained_agent))
                print(f"[DONE] agent {i+1}/{num_agents_per_beta} برای β={beta} آموزش دید.")
            except Exception as e:
                print(f"[ERROR] آموزش agent {i+1} برای β={beta} شکست خورد: {e}")

    print("[SUCCESS] آموزش تمام agentها به پایان رسید.")
    return trained_agents
