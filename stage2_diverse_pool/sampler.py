import numpy as np
from scipy.stats import norm


def compute_return_rates(df, window):
    """محاسبه نرخ بازده buy-and-hold برای تمام بازه‌های با طول مشخص"""
    prices = df['price_close'].values
    returns = []
    for i in range(len(prices) - window):
        r = (prices[i + window] - prices[i]) / prices[i]
        returns.append(r)
    return np.array(returns)


def kernel_density_estimate(rates, bandwidth=None):
    """محاسبه KDE با استفاده از توزیع نرمال و Silverman bandwidth"""
    if bandwidth is None:
        bandwidth = 1.06 * np.std(rates) * len(rates) ** (-1 / 5)

    def pdf(x):
        return np.mean(norm.pdf((x - rates[:, None]) / bandwidth), axis=0) / bandwidth

    return pdf


def biased_sampling_indices(rates, beta, theta=0.2):
    """نمونه‌گیری bias شده براساس beta مطابق معادله (4) مقاله"""
    q_low = np.quantile(rates, theta / 2)
    q_high = np.quantile(rates, 1 - theta / 2)
    print(f"[DEBUG] β={beta}, θ={theta}, q_low={q_low:.4f}, q_high={q_high:.4f}")

    pdf_est = kernel_density_estimate(rates)

    priorities = []
    for idx, r in enumerate(rates):
        if idx % 100000 == 0:
            print(f"[TRACE] بررسی نرخ بازده {idx}/{len(rates)}: r={r:.5f}")

        try:
            density = pdf_est([r])[0]
        except Exception as e:
            print(f"[ERROR] KDE شکست خورد برای r={r:.5f}: {e}")
            density = 1e-6

        if q_low <= r <= q_high:
            score = np.exp(beta * r) / max(density, 1e-6)
        else:
            score = np.exp(beta * r)

        if np.isnan(score) or np.isinf(score):
            print(f"[WARN] نمره نامعتبر برای r={r:.5f}, score={score}")
            score = 1e-6

        priorities.append(score)

    priorities = np.array(priorities)
    priorities = np.nan_to_num(priorities)
    priorities /= priorities.sum() + 1e-8
    print(f"[DEBUG] مجموع نهایی priorities: {priorities.sum():.4f}")
    return priorities
