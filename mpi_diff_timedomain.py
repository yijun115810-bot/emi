import numpy as np
from scipy.io import savemat, loadmat

# ========== 只保留采样率，删掉位移、速度、补偿距离、激励频率（不需要可以继续删） ==========
Fs = 1000000

# ==================================================================================
def data_trans(data_path, noise_path):
    # 读取dat粒子信号，*1000转mV，和原版一致
    data_array = np.fromfile(data_path, dtype=np.float64) * 1000

    # 读取噪声mat
    if noise_path.endswith(".mat"):
        mat_dict = loadmat(noise_path, squeeze_me=True)
        noise_array = mat_dict["full_signal"]
    else:
        noise_array = np.fromfile(noise_path, dtype=np.float64) * 1000

    # ========= 修改重点：不再引入read_data_num，只取两个信号长度最小值 =========
    data_array_len = len(data_array)
    noise_array_len = len(noise_array)
    data_len = min(data_array_len, noise_array_len)  # 仅两个信号长度对齐

    data_array_part = data_array[:data_len - 1]
    noise_array_part = noise_array[:data_len - 1]

    # NaN置零，和原版一致
    data_array_part[np.where(np.isnan(data_array_part))] = 0
    noise_array_part[np.where(np.isnan(noise_array_part))] = 0

    return data_array_part, noise_array_part


def signal_time_subtract_save(particle_dat, bg_mat, save_mat_path):
    sig_particle, sig_bg = data_trans(particle_dat, bg_mat)
    # 核心时域相减
    clean_signal = sig_particle - sig_bg

    # 存储字典，key沿用工程习惯full_signal
    save_info = {
        "full_signal": clean_signal,
        "Fs": Fs,
        "raw_particle_signal": sig_particle,
        "raw_background_noise": sig_bg
    }
    savemat(save_mat_path, save_info)

    # 打印核对信息
    print("=====时域相减处理完毕=====")
    print(f"粒子信号采样点数：{len(sig_particle)}")
    print(f"背景噪声采样点数：{len(sig_bg)}")
    print(f"降噪后信号采样点数：{len(clean_signal)}")
    print(f"降噪信号幅值范围：最小值 {np.min(clean_signal):.4f} mV，最大值 {np.max(clean_signal):.4f} mV")
    print(f"文件保存路径：{save_mat_path}")

    return clean_signal, save_info


if __name__ == "__main__":
    # 文件路径自行修改
    lz_dat = r'E:\20260910\1ug\lz\data2.dat'
    noise_mat = r'E:\20260910\pre_mpi_3_910\output\predict_1ug_lz.mat'
    output_mat = r'E:\20260910\pre_mpi_3_910\output\timesubstract_1ug_lz.mat'

    final_clean_sig, mat_data = signal_time_subtract_save(lz_dat, noise_mat, output_mat)
