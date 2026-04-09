import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from datetime import datetime, timedelta

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
# data = sio.loadmat('case2_comp.mat')
data = sio.loadmat('case2_error_results.mat')
alpha = data['alpha'].flatten()
beta  = data['beta'].flatten()
loss  = data['loss'].flatten()
loss[loss < 1e0] = 1e0
n = len(alpha)

# ── 构造时间轴：每分钟一个点，从 0:00 到 22:00（1321 个点）──
t0 = datetime(2026, 1, 1, 0, 0)
times = [t0 + timedelta(minutes=i) for i in range(n)]

# ── 绘图 ──
fig, ax_loss = plt.subplots(figsize=(10, 4.5))
ax_param = ax_loss.twinx()

# loss — stem（浅灰色竖线 + 深蓝色顶点）
markerline, stemlines, baseline = ax_loss.stem(
    times, loss,
    linefmt='-', markerfmt='o', basefmt=' ',
)
plt.setp(stemlines, color='#C0C0C0', linewidth=0.4)
plt.setp(markerline, color='#555555', markersize=2.5)
ln1 = plt.Line2D([], [], color='#555555', marker='o', linestyle='None',
                  markersize=3, label='Loss')

ax_loss.set_yscale('log')
ax_loss.set_ylim(1e0, 1e7)
ax_loss.set_yticks([10**i for i in range(0, 8)])
ax_loss.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))
ax_loss.set_ylabel('Loss')

# alpha, beta — stairs（阶梯图）
ln2, = ax_param.step(times, alpha, where='mid', color='#C0392B',
                      linewidth=1.5, linestyle='-', label=r'$\alpha$')
ln3, = ax_param.step(times, beta,  where='mid', color='#27AE60',
                      linewidth=1.5, linestyle='-', label=r'$\beta$')
ax_param.set_ylim(0.65, 1.0)
ax_param.set_yticks(np.arange(0.65, 1.01, 0.05))
ax_param.set_ylabel('Parameter value')

# ── 时间轴格式 ──
ax_loss.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax_loss.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax_loss.xaxis.set_minor_locator(mdates.HourLocator())
ax_loss.set_xlabel('Time')
ax_loss.set_xlim(t0, t0 + timedelta(hours=22))

# 合并图例
lines = [ln1, ln2, ln3]
labels = [l.get_label() for l in lines]
# ax_loss.legend(lines, labels, loc='upper center', ncol=3,
#                bbox_to_anchor=(0.5, 1.15), fontsize=18)

ax_loss.minorticks_on()
ax_param.minorticks_on()

fig.tight_layout()
fig.savefig('case2_convergence.svg', bbox_inches='tight')
plt.show()
