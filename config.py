# ⚙️ پارامترهای مرحله دوم EarnHFT

# طول window برای segmentها و نرخ بازده‌ها
WINDOW_LENGTH = 60  # 60 ثانیه

# لیست biasها (beta values) برای تنوع در استراتژی‌ها
BETA_LIST = [-90, -10, 30, 100]

# تعداد عامل برای هر beta
AGENTS_PER_BETA = 5

# θ برای تعیین quantileهای بالا و پایین در معادله (4)
THETA = 0.2

# تعداد برچسب روند برای segmentation
NUM_TREND_LABELS = 5

# مقادیر مجاز موقعیت اولیه
INITIAL_POSITIONS = [0]  # فقط long بدون پوزیشن

# فضای گسسته موقعیت‌ها (براساس max position)
MAX_POSITION = 1000
ACTION_SIZE = 5
ACTION_VALUES = [i * MAX_POSITION // (ACTION_SIZE - 1) for i in range(ACTION_SIZE)]

# نرخ کمیسیون
COMMISSION = 0.0002
