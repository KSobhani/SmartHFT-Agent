import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def segment_market(df, col='price_close'):
    """برش اولیه بازار بر اساس نقاط بیشینه/کمینه"""
    prices = df[col].values
    idx_max = argrelextrema(prices, np.greater)[0]
    idx_min = argrelextrema(prices, np.less)[0]
    extrema = sorted(np.concatenate([idx_max, idx_min]))

    segments = [(extrema[i], extrema[i+1]) for i in range(len(extrema)-1)]
    return segments


def merge_similar_segments(segments, df, slope_thresh=0.05, dtw_thresh=0.1, col='price_close'):
    """ادغام segmentهای با شیب و DTW مشابه"""
    merged = []
    i = 0
    while i < len(segments):
        start, end = segments[i]
        while i + 1 < len(segments):
            next_start, next_end = segments[i+1]
            seg1 = df[col].iloc[start:end].values.flatten()
            seg2 = df[col].iloc[next_start:next_end].values.flatten()

            slope1 = (seg1[-1] - seg1[0]) / len(seg1)
            slope2 = (seg2[-1] - seg2[0]) / len(seg2)
            dtw_dist, _ = fastdtw(seg1, seg2, dist=lambda x, y: abs(x - y))
            if abs(slope1 - slope2) < slope_thresh and dtw_dist / len(seg1) < dtw_thresh:
                end = next_end
                i += 1
            else:
                break
        merged.append((start, end))
        i += 1
    return merged


def label_segments(df, segments, num_labels=5):
    """برچسب‌گذاری segmentها بر اساس شیب قیمت"""
    slopes = []
    for start, end in segments:
        price_start = df['price_close'].iloc[start]  # ✅ اصلاح شد
        price_end = df['price_close'].iloc[end]      # ✅ اصلاح شد
        slope = (price_end - price_start) / (end - start)
        slopes.append(slope)

    thresholds = np.quantile(slopes, np.linspace(0, 1, num_labels + 1))
    labels = []
    for slope in slopes:
        label = np.digitize(slope, thresholds[1:-1])  # بازه‌بندی
        labels.append(label)

    # برچسب‌گذاری نهایی برای هر نقطه
    trend_label = np.zeros(len(df), dtype=int)
    for (start, end), label in zip(segments, labels):
        trend_label[start:end] = label

    df['trend_label'] = trend_label
    return df
