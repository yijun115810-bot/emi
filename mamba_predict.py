import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from mamba_ssm.modules.mamba_simple import Mamba

plt.rcParams['agg.path.chunksize'] = 10000

# 长度对齐函数（和训练代码完全一致，解决跳跃拼接维度不匹配）
def align_length(x, target_len):
    curr_len = x.size(-1)
    if curr_len == target_len:
        return x
    diff = target_len - curr_len
    pad_left = diff // 2
    pad_right = diff - pad_left
    return F.pad(x, (pad_left, pad_right))

# ====================== 和训练代码完全一致的 MambaBlock1D ======================
class MambaBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj_in = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.LayerNorm(out_channels)

        # 适配 mamba-ssm 1.0.1
        self.mamba = Mamba(
            d_model=out_channels,
            d_state=16,
            d_conv=4,
            expand=2,
            bimamba_type="v3"
        )

        self.proj_out = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.act = nn.GELU()

        self.shortcut = nn.Conv1d(in_channels, out_channels, 1, bias=False) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.proj_in(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.mamba(x)
        x = x.transpose(1, 2)
        x = self.proj_out(x)

        return self.act(x + residual)

# ====================== 和训练代码完全一致的 MambaUNet1D 四级1D Mamba-UNet ======================
class MambaUNet1D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32, depth=4):
        super().__init__()

        self.depth = depth
        self.pool = nn.MaxPool1d(2)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.enc_channels = []
        ch = base_channels
        for i in range(depth):
            self.enc_blocks.append(
                MambaBlock1D(in_channels if i == 0 else ch // 2, ch)
            )
            self.enc_channels.append(ch)
            ch *= 2

        # Bottleneck
        self.bottleneck = MambaBlock1D(ch // 2, ch)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(depth):
            in_ch = ch
            out_ch = ch // 2
            self.upconvs.append(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            self.dec_blocks.append(
                MambaBlock1D(out_ch * 2, out_ch)
            )
            ch = out_ch

        # Output
        self.final_conv = nn.Conv1d(base_channels, out_channels, 1)

    # 对齐函数
    def align(self, x, target):
        if x.size(-1) == target.size(-1):
            return x
        diff = target.size(-1) - x.size(-1)
        return F.pad(x, (diff // 2, diff - diff // 2))

    def forward(self, x):
        skips = []
        out = x

        # Encoder
        for i, block in enumerate(self.enc_blocks):
            out = block(out)
            skips.append(out)
            out = self.pool(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder
        for up, block, skip in zip(self.upconvs, self.dec_blocks, reversed(skips)):
            out = up(out)
            out = self.align(out, skip)
            out = torch.cat([out, skip], dim=1)
            out = block(out)

        return self.final_conv(out)

# ============================================================
# 2. 数据处理
# ============================================================
def _normalize_signal(signal):
    signal = signal.astype(np.float32)
    mean = np.mean(signal)
    std = np.std(signal) + 1e-8
    return (signal - mean) / std, mean, std

def _adjust_length(signal, target_length=50000):
    signal = signal.flatten()
    if len(signal) > target_length:
        start = (len(signal) - target_length) // 2
        signal = signal[start:start+target_length]
    else:
        pad_left = (target_length - len(signal)) // 2
        pad_right = target_length - len(signal) - pad_left
        signal = np.pad(signal, (pad_left, pad_right), mode='constant')
    return signal

def denormalize_signal(signal, mean, std):
    return signal * std + mean

def load_single_sample(base_dir, sub_folder, file_idx, target_length=50000):
    # 仅靠 sub_folder 控制中间层级，一行修改全局生效
    paths = {
        "emi2": os.path.join(base_dir, sub_folder, "emi2", f"emi2_{file_idx}.mat"),
        "emi3": os.path.join(base_dir, sub_folder, "emi3", f"emi3_{file_idx}.mat"),
        "emi4": os.path.join(base_dir, sub_folder, "emi4", f"emi4_{file_idx}.mat"),
        "data": os.path.join(base_dir, sub_folder, "data", f"data_{file_idx}.mat")
    }

    # 检查文件是否全部存在，缺失则返回None跳过
    for p in paths.values():
        if not os.path.exists(p):
            print(f"警告：缺失文件 {p}，跳过该样本")
            return None, None, None, None, None

    data_mat = sio.loadmat(paths["data"])
    true_signal = _adjust_length(data_mat["current_signal_data"], target_length)
    true_mean = np.mean(true_signal)
    true_std = np.std(true_signal) + 1e-8

    seg_time_data = data_mat.get("seg_time_data", np.arange(target_length))
    seg_time_data = _adjust_length(seg_time_data, target_length).reshape(-1, 1)

    emi_list = []
    raw_data = {}

    for key in ["emi2", "emi3", "emi4"]:
        mat = sio.loadmat(paths[key])
        sig = _adjust_length(mat["current_signal_data"], target_length)
        norm, _, _ = _normalize_signal(sig)
        emi_list.append(norm)
        raw_data[key] = {"raw_adjusted": sig}

    raw_data["true_current"] = {"raw_adjusted": true_signal}
    input_tensor = torch.stack([torch.tensor(s) for s in emi_list]).unsqueeze(0)
    return input_tensor, raw_data, seg_time_data, true_mean, true_std

# ============================================================
# 3. 批量预测主函数
# ============================================================
def batch_predict_all_samples():
    # ========== 核心路径配置区，后续只改这里即可 ==========
    BASE_DIR = "/media/ps/8f05279d-2d9e-4c51-90f4-009a294c3293/TBY/train_mpi_3_0910/pre_lz"
    # 切换数据集只修改这一个字符串：比如原版无嵌套就填 ""，当前bgg2就填 "bgg2"
    SUB_FOLDER = "10ugbg"
    WEIGHT_PATH = "best_mamba_24256.pth"
    OUTPUT_SAVE_DIR = "predict_output/10ugbg"
    TARGET_LENGTH = 60000
    START_NUM = 1
    END_NUM = 3033

    os.makedirs(OUTPUT_SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaUNet1D(in_channels=3, out_channels=1, base_channels=32, depth=4).to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()
    print("模型加载完成，开始批量预测...")

    for num in range(START_NUM, END_NUM + 1):
        file_idx = f"{num:05d}"
        print(f"\n正在处理样本 {file_idx} ...")

        # 传入子文件夹参数，不用改动内部路径拼接
        input_tensor, raw_data, seg_time_data, mean, std = load_single_sample(
            BASE_DIR, SUB_FOLDER, file_idx, TARGET_LENGTH
        )
        if input_tensor is None:
            continue

        input_tensor = input_tensor.to(device, dtype=torch.float32)
        with torch.no_grad():
            pred = model(input_tensor)

        pred = pred.cpu().numpy().squeeze()
        pred = denormalize_signal(pred, mean, std)

        gt = raw_data["true_current"]["raw_adjusted"]
        mae = np.mean(np.abs(pred - gt))
        rmse = np.sqrt(np.mean(np.square(pred - gt)))
        print(f"该样本 MAE: {mae:.6f} | RMSE: {rmse:.6f}")

        save_name = f"predict_{file_idx}.mat"
        save_path = os.path.join(OUTPUT_SAVE_DIR, save_name)
        sio.savemat(save_path, {
            "seg_time_data": seg_time_data,
            "current_signal_data": pred.reshape(1, -1),
            "normalize_mean": mean,
            "normalize_std": std
        })
        print(f"样本 {file_idx} 预测完成，已保存至 {save_path}")

    print("\n===== 全部样本预测完成 =====")

# ============================================================
# 4. 可视化（可选，批量模式默认关闭）
# ============================================================
def visualize(raw_data, pred):
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.title("Input EMI Signal (emi3)")
    plt.plot(raw_data["emi3"]["raw_adjusted"], label="EMI3 Input", alpha=0.7)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(2,1,2)
    plt.title("Ground Truth vs Model Prediction")
    plt.plot(raw_data["true_current"]["raw_adjusted"], label="Ground Truth", c="green", lw=1.5)
    plt.plot(pred, label="MambaUNet Prediction", c="red", alpha=0.8)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================
# 5. 程序入口
# ============================================================
if __name__ == "__main__":
    batch_predict_all_samples()
