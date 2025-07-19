import pickle

# --- خواندن متریک‌ها ---
with open("router_metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

# --- چاپ مرتب ---
print(" نتایج نهایی ارزیابی Router Agent")
print("-" * 45)
for key, value in metrics.items():
    if isinstance(value, float):
        formatted = f"{value:.2f}" if abs(value) > 1 else f"{value:.4f}"
        print(f"{key:<20}: {formatted}")
    else:
        print(f"{key:<20}: {value}")

# --- خواندن متریک‌ها ---
with open("selected_betas.pkl", "rb") as f:
    metrics = pickle.load(f)

# --- چاپ مرتب ---
print(" below:")
print("-" * 45)
for key, value in metrics.items():
    if isinstance(value, float):
        formatted = f"{value:.2f}" if abs(value) > 1 else f"{value:.4f}"
        print(f"{key:<20}: {formatted}")
    else:
        print(f"{key:<20}: {value}")
