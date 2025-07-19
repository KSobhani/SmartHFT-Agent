import numpy as np
import pandas as pd

class MarketEnv:
    def __init__(self, df, action_space, commission=0.0002):
        self.df = df.reset_index(drop=True)
        self.action_space = action_space  # discrete positions: [0, ..., max_position]
        self.commission = commission
        self.window_size = 60  # number of past steps to form state
        self.max_index = len(df) - 1
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.position = 0
        self.cash = 1_000_000
        self.done = False
        return self._get_state()

    def step(self, action_index):
        if self.current_step + 1 >= len(self.df):
            done = True
            return self._get_state(), 0.0, done, {}

        assert 0 <= action_index < len(self.action_space)
        target_position = self.action_space[action_index]
        prev_position = self.position

        # قیمت فعلی برای اعمال سفارش بازار
        price_t = self.df.loc[self.current_step, 'price_close']
        price_tp1 = self.df.loc[self.current_step + 1, 'price_close']

        # اجرای سفارش بازار برای رسیدن به موقعیت هدف
        executed_cost = self._execute_market_order(target_position - prev_position, price_t)

        self.cash -= executed_cost
        self.position = target_position

        # محاسبه reward = تغییر ارزش خالص دارایی (net asset value)
        nav_t = prev_position * price_t + self.cash + executed_cost
        nav_tp1 = self.position * price_tp1 + self.cash
        reward = nav_tp1 - nav_t

        self.current_step += 1
        self.done = self.current_step >= self.max_index - 1

        return self._get_state(), reward, self.done, {}

    def _execute_market_order(self, delta_position, price):
        if delta_position == 0:
            return 0.0
        cost = delta_position * price * (1 + self.commission)
        return cost

    def _get_state(self):
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        state_features = window.astype('float32').values.flatten()
        position_state = np.array([self.position], dtype='float32')
        return np.concatenate([state_features, position_state])