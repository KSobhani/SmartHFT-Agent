
# Multi-Agent Reinforcement Learning for High-Frequency Trading

This repository contains the implementation of a multi-agent RL system inspired by the EarnHFT framework.  
It includes hierarchical decision-making with a router agent that selects the best-performing policy based on market conditions.

---

##  Project Structure

```
├── agents/                     # Reinforcement learning agents (e.g., DDQN)
│   └── ddqn_agent.py

├── envs/                       # Market simulation environment
│   └── market_env.py

├── q_teacher/                 # Optimal Q-value computation for supervised guidance
│   ├── optimal_value.py
│   └── q_trainer.py

├── stage2_diverse_pool/       # Stage 2: Diverse agent pool creation
│   ├── config.py              # β, θ, window settings
│   ├── sampler.py             # Biased sampling via KDE
│   ├── selector.py            # Agent selection based on trend label
│   └── trend_segmentation.py  # Trend segmentation using slope + DTW

├── stage3_router/             # Stage 3: High-level policy router
│   ├── evaluate_router.py     # Evaluate router performance
│   ├── high_level_env.py      # Router's custom environment
│   ├── router_agent.py        # Router agent logic
│   └── router_trainer.py      # Training script for router agent

├── utils/                     # Utility functions and data loaders
│   └── loader.py

├── results/                   # Evaluation results and visuals
│   ├── router_report.md
│   ├── selected_betas.pkl
│   ├── trained_agents_pool.pkl
│   └── router_metrics.pkl

├── train_multiagent.py        # Main script for training agent pool
├── trainer_multiagent.py      # Alternate training script for stage 2
├── train_qteacher.py          # Supervised Q* training
├── run_router.py              # Run router with selected agents
├── eval.py                    # Evaluation and reporting
├── selecting.py               # Selection logic (stage 1–2 transition)

├── requirements.txt
└── README.md
---

## 📈 Sample Evaluation Result

📄 [View Full Report](./results/router_report.md)

| Metric              | Value       |
|---------------------|-------------|
| ✅ Average Reward    | 1.50        |
| 💰 Average Profit    | 97.05       |
| 📈 Sharpe Ratio      | 9.77        |
| 🥇 Average Win Rate  | 72.54%      |
| 💹 Total Profit      | 19,410.00   |
| 🏦 Final Balance     | 119,410.00  |
| 📊 Total Return      | 19.41%      |
| 📉 Max Drawdown      | 9.26%       |

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🧪 Running

```bash
python train_multiagent.py
python run_router.py
python eval.py
```

---

## 📚 Credits

Inspired by [EarnHFT: AAAI 2024 Paper](https://arxiv.org/abs/2401.04283)
