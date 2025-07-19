# stage3_router/evaluate_router.py

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from .high_level_env import HighLevelEnv
from .router_trainer import train_router

EPISODES = 100
INITIAL_BALANCE = 100_000

def evaluate_router(df_test, agent_pool, action_values):
    if len(df_test) < 63:
        raise ValueError("داده تست باید حداقل 63 ردیف داشته باشد.")

    print("[INFO] شروع ارزیابی router روی داده تست...")

    # نمونه‌گیری برای تعیین ابعاد state
    env_tmp = HighLevelEnv(df_test, agent_pool, action_values, INITIAL_BALANCE)
    state_dim = len(env_tmp.reset())
    print(f"[INFO] ابعاد state: {state_dim}")

    # نرمال‌سازی state
    scaler = StandardScaler()
    states = []
    for i in range(60, len(df_test)):
        window = df_test.iloc[i-60:i].values.flatten()
        states.append(np.concatenate([window, [0]]))  # فرض position=0
    scaler.fit(states)

    # ساخت محیط برای آموزش
    env_train = HighLevelEnv(df_test, agent_pool, action_values, INITIAL_BALANCE)

    print("[INFO] شروع آموزش router...")
    router, rewards_history = train_router(env_train, router=None, num_episodes=EPISODES)
    print("[INFO] آموزش router به پایان رسید.")

    # backtest نهایی
    env_back = HighLevelEnv(df_test, agent_pool, action_values, INITIAL_BALANCE)
    state = env_back.reset()
    state = scaler.transform([state])[0]
    done = False

    # متغیرهای معیار
    profits, balances, rewards = [], [], []
    wins = 0
    peak_balance = env_back.balance
    drawdowns = []

    while not done:
        action = router.select_action(state, epsilon=0.0)
        raw_next, reward, done, _ = env_back.step(action)
        state = scaler.transform([raw_next])[0]

        rewards.append(reward)
        balances.append(env_back.balance)
        profits.append(env_back.balance - INITIAL_BALANCE)
        if reward > 0: wins += 1
        peak_balance = max(peak_balance, env_back.balance)
        drawdowns.append((peak_balance - env_back.balance) / peak_balance)

    # محاسبه معیارها
    avg_reward    = np.mean(rewards)
    total_profit  = balances[-1] - INITIAL_BALANCE
    sharpe        = np.mean(rewards) / (np.std(rewards) + 1e-9) * np.sqrt(252*24*3600/60)
    win_rate      = wins / len(rewards)
    max_drawdown  = np.max(drawdowns)
    avg_profit    = np.mean(profits)
    tot_ret       = (total_profit/INITIAL_BALANCE)*100
    metrics = {
        "Average Reward": avg_reward,
        "Average Profit": avg_profit,
        "Sharpe Ratio": sharpe,
        "Average Win Rate": win_rate,
        "Total Profit": total_profit,
        "Final Balance": balances[-1],
        "Total Return": tot_ret,
        "Max Drawdown": max_drawdown,
    }

    # ذخیره متریک‌ها
    with open("router_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    print("✅ متریک‌ها در 'router_metrics.pkl' ذخیره شدند.")

    return metrics
