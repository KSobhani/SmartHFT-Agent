import numpy as np

class HighLevelEnv:
    def __init__(self, df, agent_pool, action_space, commission=0.0002, initial_balance=100000):
        self.df = df.reset_index(drop=True)
        self.balance = initial_balance
        self.agent_pool = agent_pool  # {(label, position): agent}
        self.action_space = list(agent_pool.keys())  # لیستی از (label, pos)
        self.action_map = {i: k for i, k in enumerate(self.action_space)}
        self.commission = commission
        self.max_index = len(df) - 61
        self.rewards = []
        self.positions = []
        self.balances = []
        self.actions = []
        self.reset()

    def reset(self):
        self.current_step = 60
        self.position = 0
        self.cash = self.balance
        self.balance = self.balance
        self.done = False
        self.rewards.clear()
        self.positions.clear()
        self.balances.clear()
        self.actions.clear()
        return self._get_state()

    def step(self, action_index):
        label, init_pos = self.action_map[action_index]
        agent = self.agent_pool[(label, init_pos)]

        # اجرای عامل پایین‌سطح برای 60 ثانیه
        start = self.current_step
        end = self.current_step + 60
        reward = 0
        df_slice = self.df.iloc[start:end+1].reset_index(drop=True)

        from envs.market_env import MarketEnv
        env = MarketEnv(df_slice, action_space=self._position_space(), commission=self.commission)
        env.position = self.position
        env.cash = self.cash
        self.balance = self.cash
        state = env.reset()

        for _ in range(60):
            action = agent.select_action(state, epsilon=0.0)
            next_state, r, done, _ = env.step(action)
            reward += r
            state = next_state

        self.position = env.position
        self.cash = env.cash
        price = df_slice.loc[len(df_slice)-1, 'price_close']
        self.balance = self.cash + self.position * price
        self.current_step += 60
        self.done = self.current_step + 60 >= self.max_index

        self.rewards.append(reward)
        self.positions.append(self.position)
        self.balances.append(self.balance)
        self.actions.append(action_index)

        return self._get_state(), reward, self.done, {}

    def _get_state(self):
        window = self.df.iloc[self.current_step - 60:self.current_step]
        technical = window.values.flatten()
        return np.concatenate([technical, [self.position]])

    def _position_space(self):
        return sorted(list(set([pos for (_, pos) in self.agent_pool.keys()])))

    def get_metrics(self):
        from scipy.stats import sem
        rewards = np.array(self.rewards)
        balances = np.array(self.balances)

        avg_reward = np.mean(rewards)
        avg_profit = np.mean(np.diff(balances))
        std_profit = np.std(np.diff(balances))
        sharpe = (avg_profit / std_profit) * np.sqrt(60) if std_profit > 0 else 0
        win_rate = np.mean(np.array(np.diff(balances)) > 0)
        drawdown = np.max(np.maximum.accumulate(balances) - balances)
        total_profit = balances[-1] - balances[0] if len(balances) > 1 else 0
        final_balance = balances[-1] if len(balances) > 0 else self.cash

        return {
            "Average Reward": avg_reward,
            "Average Profit": avg_profit,
            "Sharpe Ratio": sharpe,
            "Average Win Rate": win_rate,
            "Drawdown": drawdown,
            "Total Profit": total_profit,
            "Final Balance": final_balance
        }
