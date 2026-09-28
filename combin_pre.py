import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# ===================== 超参必须和MATLAB切分代码完全一致 =====================
SAMPLE_FREQ = 1000000
SEG_LENGTH = 60000
STEP_LEN = SEG_LENGTH // 2  # 25000，50%重叠
TOTAL_NUM = 3033            # 片段数量 00001 ~ 02235
BASE_NUM = 1

# 路径配置
PREDICT_FOLDER = r"E:\20260910\pre_mpi_3_910\output\500ng"
DATA_GT_FOLDER = r"E:\20260910\pre_mpi_3_910\input\500ng\data"
# 输出根目录
OUTPUT_ROOT = r"E:\20260910\pre_mpi_3_910\output"
SAVE_MAT_NAME = "predict_500ng_lz.mat"
# 完整mat保存路径
mat_save_path = os.path.join(OUTPUT_ROOT, SAVE_MAT_NAME)

# 自动创建输出文件夹
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 计算拼接后完整信号总长度
total_full_len = SEG_LENGTH + (TOTAL_NUM - 1) * STEP_LEN
print(f"完整信号总点数: {total_full_len}")

# 初始化累加数组与计数数组，用于重叠区域求平均
full_signal = np.zeros(total_full_len, dtype=np.float64)
count_map = np.zeros(total_full_len, dtype=np.int32)

# 用于全局累加所有片段绝对MAE、RMSE 以及相对百分比误差
total_mae_sum = 0.0
total_rmse_sum = 0.0
total_rel_mae_pct_sum = 0.0
total_rel_rmse_pct_sum = 0.0
valid_sample_count = 0

# 逐段读取并回填到全局数组
for seg_idx in range(1, TOTAL_NUM + 1):
    file_id = seg_idx + BASE_NUM - 1
    pred_filename = f"predict_{file_id:05d}.mat"
    gt_filename = f"data_{file_id:05d}.mat"

    pred_path = os.path.join(PREDICT_FOLDER, pred_filename)
    gt_path = os.path.join(DATA_GT_FOLDER, gt_filename)

    # 任一文件缺失则跳过本条样本
    if not os.path.exists(pred_path) or not os.path.exists(gt_path):
        print(f"警告：缺失预测文件或真值文件 {pred_filename} / {gt_filename}，跳过该片段")
        continue

    # 读取预测信号
    pred_mat = sio.loadmat(pred_path)
    pred_seg = pred_mat["current_signal_data"].flatten()

    # 读取对应data真值信号
    gt_mat = sio.loadmat(gt_path)
    gt_seg = gt_mat["current_signal_data"].flatten()

    # 计算绝对MAE、RMSE
    abs_mae = np.mean(np.abs(pred_seg - gt_seg))
    abs_rmse = np.sqrt(np.mean(np.square(pred_seg - gt_seg)))

    # 计算相对误差（除以本段真值最大幅值，转为百分比）
    gt_max_amp = np.max(np.abs(gt_seg))
    if gt_max_amp < 1e-9:
        rel_mae = 0.0
        rel_rmse = 0.0
    else:
        rel_mae = (abs_mae / gt_max_amp) * 100
        rel_rmse = (abs_rmse / gt_max_amp) * 100

    # 累加各项总和
    total_mae_sum += abs_mae
    total_rmse_sum += abs_rmse
    total_rel_mae_pct_sum += rel_mae
    total_rel_rmse_pct_sum += rel_rmse
    valid_sample_count += 1

    # 打印单样本全部指标
    print(f"样本 {file_id:05d} | 绝对MAE:{abs_mae:.6f} | 绝对RMSE:{abs_rmse:.6f} | 相对MAE:{rel_mae:.2f}% | 相对RMSE:{rel_rmse:.2f}%")

    # 原有拼接重建逻辑完全保留无修改
    start_pos = (seg_idx - 1) * STEP_LEN
    end_pos = start_pos + SEG_LENGTH
    full_signal[start_pos:end_pos] += pred_seg
    count_map[start_pos:end_pos] += 1

    if seg_idx % 100 == 0:
        print(f"===== 已处理至第 {seg_idx:05d} 片段 =====")

# 重叠区域取平均
count_map[count_map == 0] = 1
full_signal = full_signal / count_map

# 生成时间轴
full_time = np.arange(total_full_len) / SAMPLE_FREQ

# 保存拼接结果到指定文件夹
sio.savemat(mat_save_path, {
    "full_signal": full_signal,
    "full_time": full_time
})
print(f"\n拼接完成，完整信号mat文件已保存至：{mat_save_path}")

# 汇总打印整体平均误差
if valid_sample_count > 0:
    avg_abs_mae = total_mae_sum / valid_sample_count
    avg_abs_rmse = total_rmse_sum / valid_sample_count
    avg_rel_mae = total_rel_mae_pct_sum / valid_sample_count
    avg_rel_rmse = total_rel_rmse_pct_sum / valid_sample_count

    print("\n==================== 全部有效样本总平均误差 ====================")
    print(f"有效样本总数：{valid_sample_count}")
    print(f"全局平均 绝对MAE  ：{avg_abs_mae:.6f}")
    print(f"全局平均 绝对RMSE ：{avg_abs_rmse:.6f}")
    print(f"全局平均 相对MAE  ：{avg_rel_mae:.2f} %")
    print(f"全局平均 相对RMSE ：{avg_rel_rmse:.2f} %")
else:
    print("无有效配对样本，无法计算平均损失")

# 绘制拼接后波形，并保存图片到输出目录
plt.figure(figsize=(14, 5))
plt.plot(full_time, full_signal)
plt.title("Reconstructed Full Prediction Signal (Overlap Merge)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(alpha=0.3)

# 保存波形图
pic_path = os.path.join(OUTPUT_ROOT, "full_waveform.png")
plt.savefig(pic_path, dpi=300, bbox_inches="tight")
print(f"波形曲线图已保存至：{pic_path}")

plt.show()