import pandas as pd
import pickle
from utils.loader import load_data
from stage2_diverse_pool.selector import select_best_agents
from stage2_diverse_pool.config import ACTION_VALUES
from stage2_diverse_pool.trend_segmentation import segment_market, merge_similar_segments, label_segments

# --- بارگذاری داده ---
df = load_data("data/bypit_data.csv")
print("داده لود شد")
# --- ساخت سگمنت‌ها و برچسب‌گذاری روند ---
segments = segment_market(df)
merged_segments = merge_similar_segments(segments, df)
df = label_segments(df, merged_segments)

print("داده برچسب گذای شد")


# --- بارگذاری agentهای آموزش‌دیده ---
print(" شروع شد")
with open("trained_agents_pool.pkl", "rb") as f:
    trained_agents = pickle.load(f)
    
print("\n[INFO] بررسی in_features برای همه agentها:")
for beta, agent in trained_agents:
    in_dim = agent.q_net.fc1.in_features
    print(f"  β={beta}: in_features = {in_dim}")


# --- ساخت agent_pool و جمع‌آوری آمار جانبی ---
agent_pool, selection_stats, all_agent_rewards = select_best_agents(
    df, trained_agents, ACTION_VALUES, initial_positions=[0, 1, 2, 3, 4], return_stats=True
)

# --- ذخیره agent_pool ---
with open("agent_pool.pkl", "wb") as f:
    pickle.dump(agent_pool, f)
print("✅ agent_pool.pkl ذخیره شد.")

# --- ذخیره β انتخاب‌شده برای هر (label, position) ---
beta_map = { (label, pos): getattr(agent, 'beta', 'unknown') for (label, pos), agent in agent_pool.items() }
with open("selected_betas.pkl", "wb") as f:
    pickle.dump(beta_map, f)
print("✅ βهای انتخاب‌شده در selected_betas.pkl ذخیره شد.")

# --- ذخیره آمار reward برای agent منتخب ---
with open("agent_selection_stats.pkl", "wb") as f:
    pickle.dump(selection_stats, f)
print("✅ آمار reward و عملکرد agent منتخب در agent_selection_stats.pkl ذخیره شد.")

# --- ذخیره reward تمام agentها برای هر (label, pos) ---
with open("all_agent_rewards.pkl", "wb") as f:
    pickle.dump(all_agent_rewards, f)
print("✅ reward تمام agentها برای هر (label, position) در all_agent_rewards.pkl ذخیره شد.")
