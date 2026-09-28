import matplotlib.pyplot as plt
import numpy as np

# ================= 全局风格（SCI / Nature 风格） =================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ================= 数据 =================
emi_conc = np.array([0.02, 0.05, 0.1, 1, 5])
emi_val  = np.array([0.28, 0.89, 4.35, 11.68, 26.1])

trad_conc = np.array([1, 5])
trad_val  = np.array([1.18, 1.28])

noise_conc = np.array([0.02, 0.05, 0.1, 1, 5])
noise_val  = np.array([0.194, 0.39, 0.539, 0.629, 0.811])

# ================= 1. 线性坐标排布：0~0.1按真实浓度线性均分 + 0.1与1之间断轴 =================
# 左段：0.02,0.05,0.1 按原始数值线性映射到x轴位置
left_concs = np.array([0.02, 0.05, 0.1])
left_x = (left_concs - left_concs.min()) / (left_concs.max() - left_concs.min()) * 2.0
# 右段：1,5，设置一段间隙错开实现断轴
gap_width = 1.0
right_concs = np.array([1, 5])
right_x_base = left_x.max() + gap_width
right_x = right_x_base + (right_concs - right_concs.min()) / (right_concs.max() - right_concs.min()) * 1.0

# 整合全部横坐标位置
x_pos = np.concatenate([left_x, right_x])
all_tick_labels = ['0.02', '0.05', '0.1', '1', '5']

# 浓度→横坐标映射函数
def get_x(conc_arr):
    res = []
    for c in conc_arr:
        idx = np.where(emi_conc == c)[0][0]
        res.append(x_pos[idx])
    return np.array(res)

emi_x   = get_x(emi_conc)
trad_x  = get_x(trad_conc)
noise_x = get_x(noise_conc)

# ================= 排序（用于 fill_between） =================
sort_idx = np.argsort(noise_x)
noise_x_sorted = noise_x[sort_idx]
noise_val_sorted = noise_val[sort_idx]

# ================= 图像尺寸（16:10接近Nature） =================
fig, ax = plt.subplots(figsize=(8, 5.2))

# =========================================================
# Noise Floor（圆点放大 + 图例生效）
# =========================================================
ax.fill_between(
    noise_x_sorted,
    noise_val_sorted,
    np.min(noise_val_sorted) * 0.9,
    color='black',
    alpha=0.10,
    label='Noise Floor'
)

ax.plot(
    noise_x_sorted,
    noise_val_sorted,
    linestyle='--',
    color='black',
    linewidth=1.0,
    alpha=0.55
)

# Noise圆点放大，绑定图例标签
ax.scatter(
    noise_x,
    noise_val,
    color='black',
    s=35,
    zorder=2,
    label='Background Noise'
)

# =========================================================
# EMI Sensing 【修改：去除红色圆点黑色边框】
# =========================================================
ax.scatter(
    emi_x,
    emi_val,
    color='#d62728',
    s=70,
    label='EMI Sensing coil',
    zorder=3
)

ax.plot(
    emi_x,
    emi_val,
    linestyle='--',
    color='#d62728',
    linewidth=1.5,
    alpha=0.55
)

# =========================================================
# Traditional：实心蓝色圆点
# =========================================================
ax.scatter(
    trad_x,
    trad_val,
    color='#1f77b4',
    s=70,
    alpha=0.85,
    label='Conventional',
    zorder=3
)

# =========================================================
# 坐标轴刻度设置
# =========================================================
ax.set_xticks(x_pos)
ax.set_xticklabels(all_tick_labels)

# ---------- 断轴斜线符号（自动居中在0.1和1间隙） ----------
gap_mid_x = (x_pos[2] + x_pos[3]) / 2
# 归一化到ax.transAxes 0~1坐标
x_norm = (gap_mid_x - x_pos.min()) / (x_pos.max() - x_pos.min())

d = 0.012
kwargs = dict(
    transform=ax.transAxes,
    color='black',
    clip_on=False,
    linewidth=1.2
)

# 两条断轴斜线 //
ax.plot((x_norm - d, x_norm + d), (-d, +d), **kwargs)
ax.plot((x_norm + 0.025 - d, x_norm + 0.025 + d), (-d, +d), **kwargs)

# =========================================================
# 坐标轴标签
# =========================================================
ax.set_xlabel('Concentration (µg)', fontsize=12)
ax.set_ylabel('Amplitude (mV)', fontsize=12)

# =========================================================
# 网格（Nature风格淡网格）
# =========================================================
ax.grid(
    True,
    linestyle='--',
    linewidth=0.4,
    alpha=0.20
)

# =========================================================
# 图例（自动包含 Noise Floor、EMI Sensing、Traditional 三项）
# =========================================================
legend = ax.legend(
    frameon=False,
    fontsize=11,
    loc='upper left',
    handlelength=2.4,
    labelspacing=0.7,
    borderaxespad=0.4
)

# =========================================================
# 坐标轴边框
# =========================================================
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# =========================================================
# 刻度参数
# =========================================================
ax.tick_params(
    axis='both',
    which='major',
    labelsize=11,
    width=1.1,
    length=5
)

# =========================================================
# 边距优化
# =========================================================
plt.tight_layout()

# =========================================================
# 保存
# =========================================================
plt.savefig(
    r'E:\texlive\alternate_tj_latex_template_ap\f8_plot.png',
    dpi=600,
    bbox_inches='tight'
)

plt.show()