import pickle
import pandas as pd
from utils.loader import load_data
from stage2_diverse_pool.selector import select_best_agents
from stage2_diverse_pool.config import ACTION_VALUES
from stage2_diverse_pool.trend_segmentation import segment_market, merge_similar_segments, label_segments

# --- بارگذاری داده ---
print("[INFO] بارگذاری داده...")
df = load_data("data/bypit_data.csv")

# --- ساخت مجدد trend_label ---
print("[INFO] ساخت سگمنت‌های روند بازار...")
segments = segment_market(df)
print(f"[INFO] تعداد سگمنت‌های اولیه: {len(segments)}")

merged_segments = merge_similar_segments(segments, df)
print(f"[INFO] تعداد سگمنت‌ها بعد از ادغام: {len(merged_segments)}")

df = label_segments(df, merged_segments)
print("[INFO] ستون trend_label ساخته شد. مقادیر یکتا:", df['trend_label'].unique())

# --- بارگذاری عامل‌های آموزش‌دیده ---
print("[INFO] بارگذاری agentهای آموزش‌دیده...")
with open("trained_agents_pool.pkl", "rb") as f:
    trained_agents = pickle.load(f)
print(f"[INFO] تعداد agentهای آموزش‌دیده: {len(trained_agents)}")

# --- انتخاب بهترین عامل‌ها ---
print("\n🧠 شروع انتخاب بهترین agentها برای هر برچسب روند و موقعیت اولیه ...")
agent_pool = select_best_agents(df, trained_agents, ACTION_VALUES, initial_positions=[0, 1, 2, 3, 4])

# --- ذخیره agent pool نهایی ---
with open("agent_pool.pkl", "wb") as f:
    pickle.dump(agent_pool, f)

print("\n✅ استخر نهایی agentها با موفقیت ذخیره شد.")
