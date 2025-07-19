import pandas as pd
import numpy as np

def load_data(path):
    """
    بارگذاری داده و استخراج ویژگی‌های مفید مشابه با مقاله EarnHFT.
    فقط ستون‌های عددی انتخاب‌شده و در صورت نیاز ترکیبی، ساخته می‌شوند.
    """
    df = pd.read_csv(path)

    # --- حذف ستون‌های غیر عددی یا اضافی ---
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna().reset_index(drop=True)

    # --- تعریف ستون‌ها به ترتیب ---
    column_names = [
        'time',              # 0
        'price_open',        # 1
        'price_high',        # 2
        'price_low',         # 3
        'price_close',       # 4
        'taker_volume_buy',  # 5
        'taker_volume_sell', # 6
        'volume_total',      # 7
        'volume_diff',       # 8
        'liquid_vol_buy',    # 9
        'liquid_vol_sell',   # 10
        'open_interest',     # 11
        'open_interest_feat',# 12
        'spread',            # 13
        'delta_lob',         # 14
        'corr_lob_feat'      # 15
    ]
    df.columns = column_names

    # --- انتخاب ویژگی‌های مشابه EarnHFT ---
    selected = df[[
        'price_open', 'price_high', 'price_low', 'price_close',     # OHLC
        'taker_volume_buy', 'taker_volume_sell',                   # حجم خریدار/فروشنده
        'volume_diff', 'spread',                                   # شاخص فشار بازار
        'liquid_vol_buy', 'liquid_vol_sell',                       # سفارش‌های فوری
        'open_interest',                                           # موقعیت باز
        'delta_lob', 'corr_lob_feat'                               # نماینده‌های LOB
    ]].copy()

    # --- ساخت ویژگی‌های ترکیبی مفید ---
    selected['mid_price'] = (df['price_high'] + df['price_low']) / 2
    selected['imbalance'] = df['taker_volume_buy'] - df['taker_volume_sell']
    selected['liquidity_pressure'] = df['liquid_vol_buy'] - df['liquid_vol_sell']

    # --- نوع‌بندی صریح به float32 برای سازگاری با PyTorch ---
    selected = selected.astype('float32')

    return selected
