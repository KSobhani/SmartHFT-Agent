# run_router.py

import pandas as pd
import pickle
from utils.loader import load_data
from stage3_router.evaluate_router import evaluate_router
from stage2_diverse_pool.config import ACTION_VALUES

def main():
    # بارگذاری داده
    df = load_data("data/bypit_data.csv")
    test_df = df.iloc[int(len(df)*0.8):].reset_index(drop=True)

    # بارگذاری agent_pool
    with open("agent_pool.pkl", "rb") as f:
        agent_pool = pickle.load(f)

    print("[INFO] شروع ارزیابی router روی داده تست...")
    metrics = evaluate_router(test_df, agent_pool, action_values=ACTION_VALUES)
    print("✅ ارزیابی router به پایان رسید.")

if __name__ == "__main__":
    main()
