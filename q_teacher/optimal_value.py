import numpy as np

def compute_optimal_q(df, action_space, commission=0.0002):
    """
    محاسبه Q* برای هر زمان، موقعیت، و اکشن با استفاده از Dynamic Programming
    مطابق الگوریتم 1 مقاله EarnHFT.
    """
    N = len(df)
    A = action_space
    Q_star = np.zeros((N, len(A), len(A)))

    prices = df['price_close'].values  # ✅ اصلاح شد

    for t in reversed(range(N - 1)):
        for p_idx, p in enumerate(A):
            for a_idx, a in enumerate(A):
                cost = prices[t] * abs(p - a) * (1 + commission)
                profit = a * prices[t + 1]
                old_value = p * prices[t]
                Q_star[t, p_idx, a_idx] = np.max(Q_star[t + 1, a_idx]) + profit - old_value - cost

    return Q_star
