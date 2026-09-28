import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple
# 三个参数需要设置，读取原数据的文件夹目录、预测得到的mat数据和最后输出的数据名称
# 设置中文字体（避免中文乱码）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams['agg.path.chunksize'] = 10000  # 适配长信号绘制

# -------------------------- 预处理函数 --------------------------
def _normalize_signal(signal):
    """信号归一化"""
    signal = signal.astype(np.float32)
    mean = np.mean(signal)
    std = np.std(signal) + 1e-8
    return (signal - mean) / std

def _adjust_length(signal, target_length=30000):
    """调整信号长度"""
    signal = signal.flatten()
    if len(signal) > target_length:
        start = (len(signal) - target_length) // 2
        signal = signal[start:start + target_length]
    else:
        pad_left = (target_length - len(signal)) // 2
        pad_right = target_length - len(signal) - pad_left
        signal = np.pad(signal, (pad_left, pad_right), mode='constant')
    return signal

def load_mat_data(file_path: str, target_length=30000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载.mat文件并做长度调整和归一化"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")

    mat_data = sio.loadmat(file_path)
    if "seg_time_data" not in mat_data:
        raise KeyError(f"文件 {file_path} 中未找到'seg_time_data'字段")
    seg = mat_data["seg_time_data"].flatten()

    curr_fields = [k for k in mat_data.keys() if k.startswith("current_signal_data")]
    if not curr_fields:
        raise KeyError(f"文件 {file_path} 中未找到'current_signal_data'开头的字段")
    curr_raw = mat_data[curr_fields[0]].flatten()

    curr_adjusted = _adjust_length(curr_raw, target_length)
    curr_normalized = _normalize_signal(curr_adjusted)
    seg_adjusted = _adjust_length(seg, target_length)

    return seg_adjusted, curr_raw, curr_normalized

def calculate_l1_loss(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """计算归一化和原始尺度的L1损失"""
    target_length = 30000
    y_true_adjusted = _adjust_length(y_true, target_length)
    y_pred_adjusted = _adjust_length(y_pred, target_length)

    y_true_norm = _normalize_signal(y_true_adjusted)
    y_pred_norm = _normalize_signal(y_pred_adjusted)

    min_len = min(len(y_true_norm), len(y_pred_norm))
    y_true_norm = y_true_norm[:min_len]
    y_pred_norm = y_pred_norm[:min_len]

    l1_loss_norm = np.mean(np.abs(y_true_norm - y_pred_norm))
    l1_loss_raw = np.mean(np.abs(y_true_adjusted[:min_len] - y_pred_adjusted[:min_len]))

    return l1_loss_norm, l1_loss_raw

# -------------------------- 指标计算通用函数 --------------------------
def calc_rmse(y_true, y_pred):
    """计算RMSE"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calc_mape(y_true, y_pred):
    """计算MAPE(%)，防除0"""
    mask = np.abs(y_true) > 1e-10
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# -------------------------- 绘图函数 --------------------------
def plot_signal_comparison(seg_true, curr_true, seg_pred, curr_pred, save_path):
    min_len = min(len(seg_true), len(seg_pred))
    seg_true = seg_true[:min_len]
    curr_true = curr_true[:min_len]
    seg_pred = seg_pred[:min_len]
    curr_pred = curr_pred[:min_len]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(seg_true, curr_true, label="真实数据", color="red", linewidth=1.5)
    ax.plot(seg_pred, curr_pred, label="预测数据", color="blue", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_difference(seg, diff, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(seg, diff, color="green", linewidth=1.2)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlim(0, 0.03)
    ax.set_ylim(-40, 40)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_diff_to_mat(x_seg: np.ndarray, y_diff: np.ndarray, save_path: str):
    save_dict = {"x_seg": x_seg, "y_diff": y_diff}
    sio.savemat(save_path, save_dict)
    print(f"差值数据已保存至：{save_path}")

# -------------------------- 主函数 --------------------------
def main():
    # 文件路径
    base_dir_data = "predict_mps_new/bg2/data"
    base_dir_pred = "predict_mps_new"
    os.makedirs(base_dir_pred, exist_ok=True)

    true_file = os.path.join(base_dir_data, "data_001.mat")
    pred_file = os.path.join(base_dir_pred, "predict_mamba_bg1_1-300.mat")
    target_length = 30000

    compare_save_path = os.path.join(base_dir_pred, "signal_comparison.png")
    diff_save_path = os.path.join(base_dir_pred, "signal_difference.png")
    custom_diff_save_path = os.path.join(base_dir_pred, "signal_difference_custom_crop.png")

    # 加载数据
    seg_true, curr_true_raw, curr_true_norm = load_mat_data(true_file, target_length)
    seg_pred, curr_pred_raw, curr_pred_norm = load_mat_data(pred_file, target_length)
    print("数据加载完成！")

    # 全局对齐
    y_true_adj = _adjust_length(curr_true_raw, target_length)
    y_pred_adj = _adjust_length(curr_pred_raw, target_length)
    min_len = min(len(y_true_adj), len(y_pred_adj))
    y_true_full = y_true_adj[:min_len]
    y_pred_full = y_pred_adj[:min_len]
    diff = y_true_full - y_pred_full

    # ---------- 原始信号统计 ----------
    print("\n===== 原始信号统计信息 =====")
    print("---- 真实信号 ----")
    print(f"均值：{np.mean(curr_true_raw):.6f}")
    print(f"方差：{np.var(curr_true_raw):.6f}")
    print(f"标准差：{np.std(curr_true_raw):.6f}")
    print(f"最大值：{np.max(curr_true_raw):.6f}")
    print(f"最小值：{np.min(curr_true_raw):.6f}")

    print("---- 预测信号 ----")
    print(f"均值：{np.mean(curr_pred_raw):.6f}")
    print(f"方差：{np.var(curr_pred_raw):.6f}")
    print(f"标准差：{np.std(curr_pred_raw):.6f}")
    print(f"最大值：{np.max(curr_pred_raw):.6f}")
    print(f"最小值：{np.min(curr_pred_raw):.6f}")

    # 绘制信号对比
    plot_signal_comparison(seg_true, curr_true_raw, seg_pred, curr_pred_raw, compare_save_path)
    print(f"信号对比图已保存至：{compare_save_path}")

    # 全局L1、RMSE、MAPE
    l1_loss_norm, l1_loss_raw = calculate_l1_loss(curr_true_raw, curr_pred_raw)
    rmse_full = calc_rmse(y_true_full, y_pred_full)
    mape_full = calc_mape(y_true_full, y_pred_full)

    print(f"\n归一化尺度L1损失：{l1_loss_norm:.6f}")
    print(f"原始尺度L1损失：{l1_loss_raw:.6f}")
    print(f"全局整体 RMSE：{rmse_full:.6f}")
    print(f"全局整体 MAPE：{mape_full:.6f} %")

    # 绘制全局差值图
    plot_difference(seg_true[:min_len], diff, diff_save_path)
    print(f"差值图已保存至：{diff_save_path}")

    # 自定义截取差值区间
    print(f"\n===== 自定义截取差值区间 =====")
    total_len = len(diff)
    print(f"差值总长度：{total_len}")
    start_idx = int(input(f"输入起始索引 (0 ~ {total_len-1})："))
    end_idx = int(input(f"输入结束索引 (0 ~ {total_len-1})："))

    # 截取信号与差值
    seg_crop = seg_true[start_idx:end_idx]
    diff_crop = diff[start_idx:end_idx]
    y_true_crop = y_true_full[start_idx:end_idx]
    y_pred_crop = y_pred_full[start_idx:end_idx]

    plot_difference(seg_crop, diff_crop, custom_diff_save_path)
    print(f"自定义截取差值图已保存至：{custom_diff_save_path}")

    # 截取区间统计 + 新增RMSE、MAPE
    print(f"截取区间均值：{np.mean(diff_crop):.6f}")
    print(f"截取区间方差：{np.var(diff_crop):.6f}")
    print(f"截取区间标准差：{np.std(diff_crop):.6f}")
    print(f"截取区间最大值：{np.max(diff_crop):.6f}")
    print(f"截取区间最小值：{np.min(diff_crop):.6f}")
    # 新增
    rmse_crop = calc_rmse(y_true_crop, y_pred_crop)
    mape_crop = calc_mape(y_true_crop, y_pred_crop)
    print(f"截取区间 RMSE：{rmse_crop:.6f}")
    print(f"截取区间 MAPE：{mape_crop:.6f} %")

    # 保存差值mat文件
    save_file_path = "S:/02 扩展系统矩阵方法/EMI_Ma/difference_bg_1-300.mat"
    save_diff_to_mat(seg_true[:min_len], diff, save_file_path)

    print("\n分析完成！")

if __name__ == "__main__":
    main()