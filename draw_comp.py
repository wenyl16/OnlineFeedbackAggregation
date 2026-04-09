import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

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
comp   = sio.loadmat('case2_peb_error_results.mat')
result = sio.loadmat('case2_error_results.mat')

loss_comp   = comp['loss'].flatten()
loss_result = result['loss'].flatten()
loss_comp[loss_comp < 1e0] = 1e0
loss_result[loss_result < 1e0] = 1e0
n = len(loss_comp)

# ── 按小时分箱 ──
n_hours = 22
hourly_comp, hourly_result = [], []
for h in range(n_hours):
    s, e = h * 60, min((h + 1) * 60, n)
    hourly_comp.append(loss_comp[s:e])
    hourly_result.append(loss_result[s:e])

# ── 绘制 grouped box plot ──
fig, ax = plt.subplots(figsize=(14, 6))

# 每个 box 对画在 h 与 h+1 之间，Baseline 偏左，Proposed 偏右
positions_b = np.arange(n_hours) + 0.35
positions_p = np.arange(n_hours) + 0.65

bp_b = ax.boxplot(hourly_comp, positions=positions_b, widths=0.3,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(color='#1A5276', linewidth=1.5),
                   whiskerprops=dict(color='#2C5F8A', linewidth=1.0),
                   capprops=dict(color='#2C5F8A', linewidth=1.0),
                   boxprops=dict(edgecolor='#2C5F8A', linewidth=1.0))

bp_p = ax.boxplot(hourly_result, positions=positions_p, widths=0.3,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(color='#922B21', linewidth=1.5),
                   whiskerprops=dict(color='#C0392B', linewidth=1.0),
                   capprops=dict(color='#C0392B', linewidth=1.0),
                   boxprops=dict(edgecolor='#C0392B', linewidth=1.0))

for box in bp_b['boxes']:
    box.set_facecolor('#A9CCE3')
    box.set_alpha(0.6)

for box in bp_p['boxes']:
    box.set_facecolor('#F5B7B1')
    box.set_alpha(0.7)

# ── 坐标轴 ──
ax.set_xticks(np.arange(0, 23))
# ax.set_xticklabels([f'{h}:00' for h in range(0, 23)], fontsize=12, rotation=45, ha='center', rotation_mode='anchor')
ax.set_xlabel('Time')
ax.set_yscale('log')
ax.set_ylabel('Loss')
ax.set_xlim(0, n_hours)
ax.set_ylim(0.7, 1e7)
ax.set_yticks([10**i for i in range(0, 8)])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.tick_params(axis='y', which='minor', length=0)
ax.axhline(1e0, color='black', linewidth=0.5, linestyle='--', zorder=0)

# 图例
legend_elements = [
    Patch(facecolor='#A9CCE3', edgecolor='#2C5F8A', linewidth=1.2, label='Baseline'),
    Patch(facecolor='#F5B7B1', edgecolor='#C0392B', linewidth=1.2, label='Proposed'),
]
# ax.legend(handles=legend_elements, loc='upper right', fontsize=18)

ax.minorticks_on()
fig.tight_layout()
fig.savefig('case2_comparison.svg', bbox_inches='tight')
plt.show()
