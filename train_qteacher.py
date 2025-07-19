import pandas as pd
import numpy as np
import torch
from envs.market_env import MarketEnv
from q_teacher.optimal_value import compute_optimal_q
from agents.ddqn_agent import DDQNAgent
from q_teacher.q_trainer import train_with_q_teacher
from utils.loader import load_data

# --- پارامترها ---
CSV_PATH = 'data/bypit_data.csv'
COMMISSION = 0.0002
BETA_VALUES = [-90, -30, 30, 100]
MAX_POSITION = 1000
ACTION_SPACE = [0, 1, 2, 3, 4]
action_values = [i * MAX_POSITION // (len(ACTION_SPACE) - 1) for i in ACTION_SPACE]

# --- داده و محیط ---
df = load_data(CSV_PATH)
env = MarketEnv(df, action_values, commission=COMMISSION)
sample_state = env.reset()
state_dim = sample_state.shape[0]
action_dim = len(action_values)

print(f"✅ state_dim = {state_dim}, action_dim = {action_dim}")

# --- آموزش agent برای هر β و ذخیره ---
trained_agents = []
for beta in BETA_VALUES:
    print(f"🧠 آموزش agent برای β={beta} ...")
    agent = DDQNAgent(state_dim, action_dim, alpha=1.0, gamma=0.99, lr=1e-4, beta=beta)
    Q_star = compute_optimal_q(df, action_values, commission=COMMISSION)
    trained_agent = train_with_q_teacher(agent, env, Q_star, num_steps=10000, epsilon=0.1, batch_size=64, target_update=200)
    trained_agents.append((beta, trained_agent))
    print(f"✅ آموزش agent برای β={beta} تمام شد.\n")

# --- ذخیره تمام agentها ---
import pickle
with open("trained_agents_pool.pkl", "wb") as f:
    pickle.dump(trained_agents, f)

print("✅ فایل trained_agents_pool.pkl با موفقیت ذخیره شد.")

print(f"✅ مدل ذخیره شد با state_dim={state_dim}")