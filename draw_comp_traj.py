import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from scipy.ndimage import uniform_filter1d

# ── 全局字体设置（IEEE Trans 风格）──
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 20,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
})

# ── 读取数据 ──
peb_res = sio.loadmat('case2_peb_error_results.mat')
proposed_res = sio.loadmat('case2_error_results.mat')
pb_res = sio.loadmat('case2_inner_pb_results.mat')

# ── 提取 p_actual 轨迹 ──
p_actual_peb = peb_res['p_actual'].flatten()
p_actual_proposed = proposed_res['p_actual'].flatten()
p_actual_pb = pb_res['p_actual'].flatten()

# ── 计算平滑趋势和波动边界 ──
window_size = 51  # 滑动窗口大小

def compute_smooth_bounds(data, window):
    """计算波动的上下边界线"""
    # 上边界：滑动窗口内的最大值
    upper = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        upper[i] = np.max(data[start:end])

    # 下边界：滑动窗口内的最小值
    lower = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        lower[i] = np.min(data[start:end])

    # 再次平滑边界线
    upper = uniform_filter1d(upper, size=window//3, mode='mirror')
    lower = uniform_filter1d(lower, size=window//3, mode='mirror')

    return upper, lower

# 计算三条曲线的平滑边界
upper_peb, lower_peb = compute_smooth_bounds(p_actual_peb, window_size)
upper_proposed, lower_proposed = compute_smooth_bounds(p_actual_proposed, window_size)
upper_pb, lower_pb = compute_smooth_bounds(p_actual_pb, window_size)

# ── 时间轴格式化 ──
def time_formatter(x, pos):
    total_minutes = int(x)
    hours = (total_minutes // 60) % 24
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"

# ── 绘图 ──
fig, ax = plt.subplots(figsize=(14, 7))

time_steps = np.arange(len(p_actual_peb))

# 绘制阴影区域（波动范围）
ax.fill_between(time_steps, lower_peb, upper_peb, color='#1f77b4', alpha=0.15, linewidth=0)
ax.fill_between(time_steps, lower_proposed, upper_proposed, color='#ff7f0e', alpha=0.15, linewidth=0)
ax.fill_between(time_steps, lower_pb, upper_pb, color='#2ca02c', alpha=0.15, linewidth=0)

# 绘制原始轨迹线（浅色，在阴影区域内）
ax.plot(time_steps, p_actual_peb, color='#1f77b4', linewidth=0.3, alpha=0.3)
ax.plot(time_steps, p_actual_proposed, color='#ff7f0e', linewidth=0.3, alpha=0.3)
ax.plot(time_steps, p_actual_pb, color='#2ca02c', linewidth=0.3, alpha=0.3)

# 绘制上下边界线（实线，更平滑）
ax.plot(time_steps, upper_peb, color='#1f77b4', linewidth=2.0, alpha=0.95)
ax.plot(time_steps, lower_peb, color='#1f77b4', linewidth=2.0, alpha=0.95, label='PEB')

ax.plot(time_steps, upper_proposed, color='#ff7f0e', linewidth=2.0, alpha=0.95)
ax.plot(time_steps, lower_proposed, color='#ff7f0e', linewidth=2.0, alpha=0.95, label='Proposed')

ax.plot(time_steps, upper_pb, color='#2ca02c', linewidth=2.0, alpha=0.95)
ax.plot(time_steps, lower_pb, color='#2ca02c', linewidth=2.0, alpha=0.95, label='Inner Boundary')

ax.set_xlabel('Time (HH:MM)')
ax.set_ylabel('Actual Power (kW)')
# ax.legend(loc='upper right')
# ax.grid(True, alpha=0.3)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
ax.xaxis.set_major_locator(ticker.MultipleLocator(120))

ax.set_xlim(0, len(time_steps) - 1)
ax.set_ylim(-1000, 1250)

plt.tight_layout()
plt.savefig('Fig_Comparison_Trajectory.svg', dpi=300, bbox_inches='tight')
plt.show()
