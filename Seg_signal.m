clc
clear
close all

% 设置参数
% 对数据进行裁剪，可以先看到数据的样子，然后选择裁剪路径和编号段落
Sample_Frequency = 1000000;   % [Hz] 采样频率
dat_read_path = 'S:\02 扩展系统矩阵方法\2026.6.3\2026.6.3\LZ';  % .dat文件读取路径
file_num_base =                                         1;  % 基础编号（如29→029，1→001）
segment_length = 30000;  % 每段固定点数（与原代码保持一致）
% 定义.dat文件列表和信号映射
dat_files = {'data1.dat', 'data2.dat', 'data4.dat', 'data5.dat'};
% 信号名称映射函数（与原代码保持一致）
signal_names = {'data', 'emi2', 'emi3', 'emi4'};
signal_data = cell(1, 4);
folder_paths = cell(1, 4);
% 步骤1：读取所有.dat文件数据
for dat_idx = 1:length(dat_files)
    dat_file = dat_files{dat_idx};
    dat_fullpath = fullfile(dat_read_path, dat_file);
    save_prefix = signal_names{dat_idx};  % 直接使用预定义的信号名称
    
    % 读取.dat文件
    try
        raw_data = read_dat_l(dat_fullpath);
        % fprintf('成功读取：%s，数据长度：%d\n', dat_file, length(raw_data));     
        % 存储到信号数据cell数组中
        signal_data{dat_idx} = raw_data;
        
    catch err
        warning('读取%s失败：%s，跳过！', dat_file, err.message);
        continue;
    end
end

% 检查是否所有数据都成功读取
valid_signals = find(~cellfun(@isempty, signal_data));
if length(valid_signals) < 4
    error('部分数据读取失败，仅成功读取 %d 个信号', length(valid_signals));
end

% 使用第一个信号的生成时间轴（假设所有信号长度相同）
signal_length = length(signal_data{1});
time_data = (0:signal_length-1)' / Sample_Frequency;  % 生成时间轴

% fprintf('数据信息:\n');
% for i = 1:4
%     fprintf('%s 长度: %d\n', signal_names{i}, length(signal_data{i}));
% end
% fprintf('time_data 范围: %.3f ~ %.3f 秒\n', time_data(1), time_data(end));

% 绘制信号图（保持原代码不变）


figure(1)
for i = 1:4
    subplot(2,2,i)
    plot(time_data, signal_data{i});
    title(['信号', num2str(i), ': ', signal_names{i}, '_plt']);
    xlabel('时间 (s)'); ylabel('幅值');
end

% 设置剪切参数（保持原代码逻辑）
signal_length = length(signal_data{1});
num_segments = floor(signal_length / segment_length);  % 计算需要多少段

fprintf('\n剪切参数:\n');
fprintf('  信号总长度: %d\n', signal_length);
fprintf('  剪切份数: %d\n', num_segments);
fprintf('  每段长度: %d\n', segment_length);

% 创建保存文件夹
for i = 1:4
    folder_paths{i} = fullfile('S:\02 扩展系统矩阵方法\EMI_DL\predict_mps_new\lz2', signal_names{i});
    % 确保文件夹存在
    if ~exist(folder_paths{i}, 'dir')
        mkdir(folder_paths{i});
    end
end

% 剪切并保存信号（保持原代码逻辑）
for seg_num = 1:num_segments
    % 计算当前段的索引范围
    start_idx = (seg_num - 1) * segment_length + 1;
    end_idx = seg_num * segment_length;
    
    % 确保最后一段包含所有剩余数据
    if seg_num == num_segments
        end_idx = signal_length;
    end
    % 可以加判断确保时间轴一直为0开头
    seg_time_data = time_data(start_idx:end_idx);
    % 对每个信号分别保存
    for sig_num = 1:4
        % 提取当前信号当前段的数据
        current_signal_data = signal_data{sig_num}(start_idx:end_idx);
        % 按照要求命名：前4个字母_切片起始位置，使用3位数字编号
        file_num = seg_num + file_num_base -1;  % 从+324开始编号（保持原代码逻辑）
        filename = fullfile(folder_paths{sig_num}, sprintf('%s_%04d.mat', signal_names{sig_num}(1:4), file_num));
        % 保存单个信号段
        save(filename, 'current_signal_data', 'seg_time_data');
        fprintf('  保存: %s_%03d.mat (长度: %d, 时间: %.3f-%.3fs)\n', ...
                signal_names{sig_num}(1:4), file_num, length(current_signal_data), ...
                seg_time_data(1), seg_time_data(end));
    end
end
fprintf('\n所有信号处理完成！\n');