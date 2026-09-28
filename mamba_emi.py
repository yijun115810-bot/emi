import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from scipy.io import loadmat
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import time

# 官方加速Mamba导入
from mamba_ssm.modules.mamba_simple import Mamba

plt.rcParams['agg.path.chunksize'] = 10000
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# 解决上采样与跳跃连接长度不一致（根治维度报错）
def align_length(x, target_len):
    curr_len = x.size(-1)
    if curr_len == target_len:
        return x
    diff = target_len - curr_len
    pad_left = diff // 2
    pad_right = diff - pad_left
    return F.pad(x, (pad_left, pad_right))

# ====================== Mamba Block【关键修改：适配mamba-ssm 1.0.1旧版参数】 ======================
class MambaBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj_in = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.LayerNorm(out_channels)

        # 严格适配 mamba-ssm 1.0.1 版本要求
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


# ====================== Mamba-UNet 1D（改进版四级编解码结构，完全不变） ======================
class MambaUNet1D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32, depth=4):
        super().__init__()

        self.depth = depth
        self.pool = nn.MaxPool1d(2)

        # ================= Encoder =================
        self.enc_blocks = nn.ModuleList()
        self.enc_channels = []

        ch = base_channels
        for i in range(depth):
            self.enc_blocks.append(
                MambaBlock1D(in_channels if i == 0 else ch // 2, ch)
            )
            self.enc_channels.append(ch)
            ch *= 2  # 通道逐层翻倍（标准UNet）

        # ================= Bottleneck =================
        self.bottleneck = MambaBlock1D(ch // 2, ch)

        # ================= Decoder =================
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i in range(depth):
            in_ch = ch
            out_ch = ch // 2

            self.upconvs.append(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
            )

            # skip + upsample concat -> 2*out_ch
            self.dec_blocks.append(
                MambaBlock1D(out_ch * 2, out_ch)
            )

            ch = out_ch

        # ================= Output =================
        self.final_conv = nn.Conv1d(base_channels, out_channels, 1)

    # ---------- 对齐 ----------
    def align(self, x, target):
        if x.size(-1) == target.size(-1):
            return x
        diff = target.size(-1) - x.size(-1)
        return F.pad(x, (diff // 2, diff - diff // 2))

    def forward(self, x):

        skips = []
        out = x

        # ================= Encoder =================
        for i, block in enumerate(self.enc_blocks):
            out = block(out)
            skips.append(out)
            out = self.pool(out)

        # ================= Bottleneck =================
        out = self.bottleneck(out)

        # ================= Decoder =================
        for up, block, skip in zip(self.upconvs, self.dec_blocks, reversed(skips)):

            out = up(out)
            out = self.align(out, skip)

            out = torch.cat([out, skip], dim=1)
            out = block(out)

        return self.final_conv(out)

# ====================== 预加载数据集（消除反复读取mat磁盘IO卡顿） ======================
class EMIDataset1D(Dataset):
    def __init__(self, data_dir='/media/ps/8f05279d-2d9e-4c51-90f4-009a294c3293/TBY/train_mpi_3_0910', target_length=60000):
        self.data_dir = data_dir
        self.target_length = target_length
        self.data_folder = os.path.join(data_dir, 'data')
        self.emi2_folder = os.path.join(data_dir, 'emi2')
        self.emi3_folder = os.path.join(data_dir, 'emi3')
        self.emi4_folder = os.path.join(data_dir, 'emi4')
        self.file_pairs = []
        self.cache = []

        for i in range(1, 24000):
            file_suffix = f"{i:05d}"
            files = [
                os.path.join(folder, f"{name}_{file_suffix}.mat")
                for folder, name in zip(
                    [self.data_folder, self.emi2_folder, self.emi3_folder, self.emi4_folder],
                    ['data', 'emi2', 'emi3', 'emi4']
                )
            ]
            if all(os.path.exists(f) for f in files):
                self.file_pairs.append(dict(zip(['data', 'emi2', 'emi3', 'emi4'], files)))

        if not self.file_pairs:
            raise ValueError("No valid data pairs found!")

        print("Preloading all dataset into memory...")
        for idx in range(len(self.file_pairs)):
            self.cache.append(self._load_single(idx))
        print("Dataset preload finished.")

    def _normalize_signal(self, signal):
        signal = signal.astype(np.float32)
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    def _adjust_length(self, signal):
        signal = signal.flatten()
        if len(signal) > self.target_length:
            start = (len(signal) - self.target_length) // 2
            return signal[start:start + self.target_length]
        else:
            pad_left = (self.target_length - len(signal)) // 2
            pad_right = self.target_length - len(signal) - pad_left
            return np.pad(signal, (pad_left, pad_right), mode='constant')

    def _load_single(self, idx):
        f = self.file_pairs[idx]
        data = loadmat(f['data'])['current_signal_data']
        emi2 = loadmat(f['emi2'])['current_signal_data']
        emi3 = loadmat(f['emi3'])['current_signal_data']
        emi4 = loadmat(f['emi4'])['current_signal_data']

        data = torch.tensor(self._normalize_signal(self._adjust_length(data))).unsqueeze(0)
        input_tensor = torch.stack([
            torch.tensor(self._normalize_signal(self._adjust_length(emi2))),
            torch.tensor(self._normalize_signal(self._adjust_length(emi3))),
            torch.tensor(self._normalize_signal(self._adjust_length(emi4)))
        ])
        return input_tensor, data

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        return self.cache[idx]

# ====================== 可视化函数 ======================
def visualize_prediction(inputs, targets, outputs, epoch):
    input_np = inputs.cpu().detach().numpy()
    target_np = targets.cpu().detach().numpy()
    output_np = outputs.cpu().detach().numpy()
    input_sample = input_np[0, :, :].squeeze()
    target_sample = target_np[0, :].squeeze()
    output_sample = output_np[0, :].squeeze()

    plt.figure(figsize=(15, 6))
    channel_names = ['emi2', 'emi3', 'emi4']
    for i in range(input_sample.shape[0]):
        plt.plot(input_sample[i], label=channel_names[i], alpha=0.7)
    plt.plot(target_sample, label='Target', color='green', linewidth=2)
    plt.plot(output_sample, label='Predicted', color='red', alpha=0.8)
    plt.title(f'Epoch {epoch+1}')
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs('prediction_plots_mamba', exist_ok=True)
    plt.savefig(f'prediction_plots_mamba/epoch_{epoch+1}.png', dpi=300)
    plt.close()

def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('loss_curve_mamba.png', dpi=300)
    plt.show()

# ====================== 训练入口（AMP半精度加速，最优训练效率） ======================
def train_mamba_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = EMIDataset1D(data_dir='/media/ps/8f05279d-2d9e-4c51-90f4-009a294c3293/TBY/train_mpi_3_0910', target_length=60000)
    print(len(dataset))

    train_size = max(7000, len(dataset) - 7000)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 超长序列显存紧张默认batch=1，显存充裕可改为2
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )

    model = MambaUNet1D(in_channels=3, out_channels=1, base_channels=32, depth=4).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()

    num_epochs = 400
    patience = 30
    patience_counter = 0
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast(dtype=torch.float16):
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining = (elapsed / (epoch + 1)) * (num_epochs - epoch - 1)
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} "
            f"| Epoch Time: {epoch_time:.1f}s | Elapsed: {elapsed/60:.1f}min | Est. Remaining: {remaining/60:.1f}min"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_mamba_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            inputs, targets = next(iter(val_loader))
            visualize_prediction(inputs, targets, model(inputs.to(device)), epoch)

    plot_loss_curve(train_losses, val_losses)
    total_training_time = time.time() - start_time
    print(f"Training completed! Best val loss: {best_val_loss:.6f}")
    print(f"Total training time: {total_training_time/60:.1f} minutes")
    return model

if __name__ == "__main__":
    trained_model = train_mamba_model()
