#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理脚本 - 将dataValues转换为点位坐标
接收传感器数据数组，返回处理后的x,y坐标
支持参数和文件两种传输模式
"""
import pandas as pd
import sys
import json
import os

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt
from scipy.linalg import eig as scipy_eig

###############################################################！！！！！步态识别定位！！！！！#######################################################################
"""
走廊实时定位处理函数

包含三个核心函数：
1. find_coordinates - 在索引表中根据距离查找坐标
2. find_peak_with_threshold - 基于阈值的峰值检测
3. process_hallway_data - 步态定位主处理函数
"""
def find_coordinates(distance, index_data):
    """根据距离在索引表中查找坐标"""
    import bisect
    distances = index_data['distances']
    x_coords = index_data['x_coords']
    y_coords = index_data['y_coords']
    
    idx = bisect.bisect_left(distances, distance)
    if idx == 0:
        return x_coords[0], y_coords[0]
    elif idx == len(distances):
        return x_coords[-1], y_coords[-1]
    else:
        # 线性插值
        dist_prev = distances[idx-1]
        dist_next = distances[idx]
        ratio = (distance - dist_prev) / (dist_next - dist_prev)
        x = x_coords[idx-1] + (x_coords[idx] - x_coords[idx-1]) * ratio
        y = y_coords[idx-1] + (y_coords[idx] - y_coords[idx-1]) * ratio
        return x, y

def find_peak_with_threshold(data_segment, segment_name):
    """在数据段中寻找有效峰值"""
    import numpy as np
    if not data_segment:
        return None, "空数据段"
        
    data_array = np.array(data_segment)
    mean_val = np.mean(data_array)
    std_val = np.std(data_array)
    
    # 设置阈值：均值 + 1.5倍标准差
    threshold = 3.5 * mean_val
    
    
    max_index = np.argmax(data_array)
    max_value = data_array[max_index]
    
    if max_value < threshold:
        return None, f"{segment_name}峰值{max_value:.3f}低于阈值{threshold:.3f},均值{mean_val:.3f}"
    
    return max_index, "有效峰值"

def process_hallway_data(data_values: List[float], index_data: Dict) -> Dict[str, float]:
    """
    处理走廊传感器数据，返回点位坐标
    
    Args:
        data_values: 传感器方差数据数组
        index_data: 索引表数据字典 {'distances': [], 'x_coords': [], 'y_coords': []}
        
    Returns:
        包含前后两段坐标的字典
    """
    try:
        # 确保所有值都是数字
        data_values = [float(val) for val in data_values]
        
        # 将数据分为前后两段
        split_index = int(790 / 0.625)  # 作为前半段和后半段的分界线
        split_start = int(217/0.625)    # 由于存在死区，因此前面217m位置处进行截断处理
        split_end   = int(970/0.625)    # 后续位置不再是关注的位置点
        
        front_data = data_values[split_start:split_index]   # 从217----790m位置为前半段区域
        back_data = data_values[split_index:split_end]      # 从790----970m位置为后半段区域
        
        # 处理前段数据 (217-790米)
        front_max_index, front_status = find_peak_with_threshold(front_data, "前段")
        if front_max_index is None:
            print(f"前段数据无有效峰值: {front_status}")
            return {"error": f"前段数据无有效峰值: {front_status}", "mode": "no_peak_detected"}
        
        front_position = front_max_index * 0.625 
        front_distance = round(front_position, 1) + 217
        front_x, front_y = find_coordinates(front_distance, index_data)
        
        # 处理后段数据 (790米-970米)
        back_max_index, back_status = find_peak_with_threshold(back_data, "后段")
        if back_max_index is None:
            print(f"后段数据无有效峰值: {back_status}")
            return {"error": f"后段数据无有效峰值: {back_status}", "mode": "no_peak_detected"}
        
        actual_back_index = split_index + back_max_index
        back_position = actual_back_index * 0.625
        back_distance = round(back_position, 1)
        back_x, back_y = find_coordinates(back_distance, index_data)
        
        print(f"前段: 索引{front_max_index}, 距离{front_distance}m, 坐标({front_x:.3f}, {front_y:.3f})")
        print(f"后段: 索引{actual_back_index}, 距离{back_distance}m, 坐标({back_x:.3f}, {back_y:.3f})")

        # 只有前半段和后半段都能找到峰值，才能定位到[x,y],不然该数据不可信
        # 将其进行合理数据融合，按照权重进行使用
        weight = 0.9
        True_x = weight * back_x + (1 - weight) * front_x
        True_y = weight * front_y + (1 - weight) * back_y 
        
        # 返回结果
        return {
            "front_x": round(front_x, 3),
            "front_y": round(front_y, 3),
            "front_distance": front_distance,
            "front_index": front_max_index,
            "back_x": round(back_x, 3),
            "back_y": round(back_y, 3),
            "back_distance": back_distance,
            "back_index": actual_back_index,
            "mode": "dual_segment_lookup",
            "True_x": round(True_x, 3),
            "True_y": round(True_y, 3),
        }
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}", "mode": "fallback"}


###############################################################！！！！！单频声源定位！！！！！#######################################################################
"""
单频声源定位处理函数
包括两个核心函数：
1. music_calc_2d_far - 在一字阵列里实现方位角度估计，范围0° - 180°
2. process_single_source_data - 单频声源定位主处理函数
"""

"""
MUSIC算法计算二维远场信号到达角(DOA)
Parameters:
-----------
signals_array : np.ndarray 输入信号阵列，形状为 (阵元数, 采样点数)    
psd_freq : float 信号中心频率，单位Hz
c : float 介质中的声速，单位m/s
psd_fft_point : int FFT点数，用于计算协方差矩阵
arrays_d : float 阵元间距，单位m
source_num : int 信源数量，需要估计的信号源个数
Returns:
--------
angles : np.ndarray 遍历方位角估计结果
spectrum_normalized : np.ndarray 在所有扫描角度上的空间谱分布
theta_estimate : float 估计的声源角度
best_steering_vector : np.ndarray 最优导向矢量
"""

def music_calc_2d_far(signals_array: np.ndarray, 
                      psd_freq: float, 
                      c: float, 
                      psd_fft_point: int, 
                      arrays_d: float, 
                      source_num: int) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    MUSIC波束定位算法
    """
    import matplotlib.pyplot as plt
    
    # 参数初始化 - 输入是 [array_num, signal_length]
    array_num, signal_length = signals_array.shape
    f = psd_freq
    
    print(f"输入信号维度: {signals_array.shape}")
    print(f"阵列数量: {array_num}, 信号长度: {signal_length}")
    
    
    # FFT变换 - 对每个通道的时间信号做FFT
    psd_signal = np.fft.fft(signals_array, psd_fft_point, axis=1)  # [psd_fft_point, array_num]
    # 取单边谱
    psd_signal_one = psd_signal[:,:psd_fft_point//2 + 1]  # [psd_fft_point//2+1, array_num]
    # 求取协方差矩阵
    R = (psd_signal_one.conj() @ psd_signal_one.T) / signal_length
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(R)
    # 排序：从小到大然后翻转（与MATLAB一致）
    sorted_indices = np.argsort(eigenvalues)  # 从小到大
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    eigenvectors_sorted = np.fliplr(eigenvectors_sorted)  # 翻转后为从大到小

    # 噪声子空间
    Un = eigenvectors_sorted[:, source_num:]
    # 计算系数
    ideal_spacing = 0.5 * c / f
    xishu = ideal_spacing / arrays_d if arrays_d != ideal_spacing else 1.0
    
    # 角度遍历
    angle_step = 1.0
    angles = np.arange(0, 181, angle_step)
    
    spectrum = np.zeros(len(angles))
    best_steering_vector = None
    max_spectrum_value = -np.inf
    
    # 预计算常数
    k = 2 * np.pi * f    # 公式 2*pi*f
    
    for i, theta in enumerate(angles):
        # 生成导向矢量
        steering_vector = np.zeros(array_num, dtype=complex)
        for j in range(array_num):
            distance = arrays_d * j * xishu
            phase = k * distance * np.cos(np.deg2rad(theta)) / c
            steering_vector[j] = np.exp(1j * phase)           
        
        # MUSIC谱计算
        temp = steering_vector.conj().T @ Un
        temp2 = temp @ Un.conj().T
        denominator = np.linalg.norm(temp2 @ steering_vector)
        spectrum[i] = 1 / (denominator + 1e-12)
        
        if spectrum[i] > max_spectrum_value:
            max_spectrum_value = spectrum[i]
            best_steering_vector = steering_vector.copy()
    
    # 峰值检测 - 判断峰值是否明显
    spectrum_normalized = spectrum / (np.max(spectrum) + 1e-12)
    max_val = np.max(spectrum_normalized)
    mean_val = np.mean(spectrum_normalized)
    std_val = np.std(spectrum_normalized)
    
    # 设置峰值信噪比比值
    peak_significance = (max_val - mean_val) / (std_val + 1e-12)
    
    # 检查峰值是否足够尖锐（通过二阶差分判断）
    spectrum_diff = np.diff(spectrum_normalized, 2)
    peak_sharpness = np.max(np.abs(spectrum_diff)) / (np.std(spectrum_diff) + 1e-12)
      
    if peak_significance < 3.0 or peak_sharpness < 2.0:
        # 峰值不明显，定位失败
        print("MUSIC定位失败：峰值不明显或不够尖锐")
        theta_estimate = 0.0
        peak_index = -1
    else:
        # 峰值明显，进行定位
        peak_index = np.argmax(spectrum_normalized)
        theta_estimate = angles[peak_index]
        print(f'峰值索引: {peak_index}')
        print(f'估计角度: {theta_estimate:.2f}°')
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    plt.plot(angles, spectrum_normalized, 'b-', linewidth=1.5, label='MUSIC谱')
    
    plt.xlabel('角度 (度)', fontsize=12)
    plt.ylabel('归一化功率谱密度', fontsize=12)
    plt.title('MUSIC算法角度估计结果', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 180)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()
    
    return angles, spectrum_normalized, theta_estimate, best_steering_vector

def process_single_source_data(data_values: List[float], data_order: int = 0, ArrayNum: int = 4) -> Dict[str, Any]:
    """
    处理单频声源数据，使用MUSIC算法进行3D定位
    
    Args:
        data_values: 传感器数据数组
        data_order: 数据顺序编号
        ArrayNum: 阵元数量
    
    Returns:
        包含角度估计和定位结果的字典
    """
    try:
        import numpy as np
        
        if not data_values:
            return {"error": "空数据数组", "mode": "fallback"}
        
        # 确保所有值都是数字
        try:
            data_values = [float(val) for val in data_values]
        except ValueError as ve:
            return {"error": f"无效数据值: {str(ve)}", "mode": "fallback"}
        
        # 将数据等分成ArrayNum份，模拟阵列信号
        arrays = np.array_split(data_values, ArrayNum)
        
        # 构建阵列信号矩阵 [signal_length, array_num]
        min_length = min(len(arr) for arr in arrays)
        signals_array = np.zeros((ArrayNum, min_length))
        
        for i in range(ArrayNum):
            signals_array[i, :] = arrays[i][:min_length]
        
        # MUSIC算法参数
        psd_freq = 1000  # 信号频率 1kHz
        c = 340          # 声速 340m/s
        psd_fft_point = 1024  # FFT点数
        arrays_d = 0.17  # 阵元间距 0.17m
        source_num = 1   # 声源数量
        
        # 调用MUSIC算法
        angles, spectrum, theta_estimate, steering_vector = music_calc_2d_far(
            signals_array=signals_array,
            psd_freq=psd_freq,
            c=c,
            psd_fft_point=psd_fft_point,
            arrays_d=arrays_d,
            source_num=source_num
        )
        
        # 检查定位是否成功
        if theta_estimate == 0.0:
            print("MUSIC定位失败：无法检测到明显的信号源")
            return {
                "error": "定位失败，无法检测到明显的信号源",
                "mode": "fallback",
                "array_num": ArrayNum,
                "signal_length": min_length
            }
        
        # 基于估计角度计算3D坐标（这里简化为2D到3D的映射）
        # 在实际应用中，可能需要多个阵列或额外的传感器来获取完整的3D坐标
        distance_estimate = 5.0  # 假设距离为5米，实际应用中需要根据信号强度或其他方法估计
        
        # 将极坐标转换为直角坐标
        theta_rad = np.deg2rad(theta_estimate)
        x = distance_estimate * np.sin(theta_rad)
        y = 2.0  # 假设高度为2米
        z = distance_estimate * np.cos(theta_rad)
        
        print(f"定位结果:")
        print(f"  估计角度: {theta_estimate:.2f}°")
        print(f"  3D坐标: ({x:.3f}, {y:.3f}, {z:.3f})")
        print(f"  阵元数量: {ArrayNum}")
        print(f"  信号长度: {min_length}")
        
        return {
            "x": round(x, 3),
            "y": round(y, 3), 
            "z": round(z, 3),
            "theta_estimate": round(theta_estimate, 2),
            "distance_estimate": round(distance_estimate, 2),
            "array_num": ArrayNum,
            "signal_length": min_length,
            "mode": "music_algorithm",
            "steering_vector_real": [round(val.real, 4) for val in steering_vector],
            "steering_vector_imag": [round(val.imag, 4) for val in steering_vector]
        }
        
    except Exception as e:
        return {
            "error": f"无人机数据处理失败: {str(e)}",
            "mode": "error_fallback"
        }

###############################################################！！！！！无人机声源定位！！！！！#######################################################################
"""
无人机声源定位处理函数
包括两个核心函数：
1. estimate_doa_3d_tetrahedral - 基于TDOA的无人机声源定位函数
2. gcc_phat-互相关函数实现
3. process_single_source_data - 单频声源定位主处理函数
"""

def estimate_doa_3d_tetrahedral(received_signals: np.ndarray, 
                               fs: float, 
                               array_pos: np.ndarray, 
                               resolution: float = 1.0) -> Tuple[float, float, np.ndarray]:
    """
    基于TDOA的3D声源定位（四面体阵列）
    
    Args:
        received_signals: 接收信号 [num_mics, signal_length]
        fs: 采样率 (Hz)
        array_pos: 麦克风位置 [3, num_mics]
        resolution: 角度分辨率 (度)
        
    Returns:
        azimuth: 方位角 (度)
        elevation: 俯仰角 (度)
        srp_map: SRP能量图
    """
    c = 343.0  # 声速 m/s
    num_mics = array_pos.shape[1]
    
    if array_pos.shape[0] != 3 or num_mics != 4:
        raise ValueError("array_pos必须是3×4矩阵，每列代表一个麦克风的[x,y,z]坐标")
    
    # 创建角度网格 (0:180°方位角, 0:90°俯仰角)
    theta_range = np.arange(0, 181, resolution)  # 方位角
    phi_range = np.arange(0, 91, resolution)     # 俯仰角
    theta_grid, phi_grid = np.meshgrid(theta_range, phi_range)
    
    # 计算GCC-PHAT
    gcc_results = {}
    for i in range(num_mics):
        for j in range(i + 1, num_mics):
            gcc, lags = gcc_general(received_signals[i], received_signals[j], fs, 'phat')
            # 翻转数据（对应MATLAB的flip）
            gcc = np.flip(gcc)
            # 归一化（使最大绝对值=1）
            gcc = gcc / (np.max(np.abs(gcc)) + 1e-12)
            gcc_results[(i, j)] = {'gcc': gcc, 'lags': lags}
    
    # SRP计算
    srp = np.zeros_like(theta_grid, dtype=float)
    
    for idx in range(theta_grid.size):
        i, j = np.unravel_index(idx, theta_grid.shape)
        theta = theta_grid[i, j]
        phi = phi_grid[i, j]
        
        # 单位方向向量
        u = np.array([
            np.cos(np.radians(theta)) * np.cos(np.radians(phi)),
            np.sin(np.radians(theta)) * np.cos(np.radians(phi)),
            np.sin(np.radians(phi))
        ])
        
        energy = 0.0
        for (mic_i, mic_j), gcc_data in gcc_results.items():
            # 计算时延
            baseline = array_pos[:, mic_i] - array_pos[:, mic_j]
            tau_ij = np.dot(u, baseline) / c
            
            # 在GCC中找到最接近的时延
            lag_idx = np.argmin(np.abs(gcc_data['lags'] - tau_ij))
            
            if np.abs(gcc_data['lags'][lag_idx] - tau_ij) < 1.0 / fs:
                energy += np.abs(gcc_data['gcc'][lag_idx])
        
        srp[i, j] = energy
    
    # 后处理
    srp_map = gaussian_filter(srp, sigma=1.0)  # 高斯平滑
    # 归一化SRP地图（使最大值为1）
    srp_map = srp_map / (np.max(srp_map) + 1e-12)

    # 峰值检测 - 判断峰值是否明显
    max_val = np.max(srp_map)
    mean_val = np.mean(srp_map)
    std_val = np.std(srp_map)

    # 设置峰值显著性阈值
    peak_significance = (max_val - mean_val) / (std_val + 1e-12)

    if peak_significance < 2.0:
        # 峰值不明显，定位失败
        azimuth = 0.0
        elevation = 0.0
    else:
        # 峰值明显，进行定位
        max_idx = np.argmax(srp_map)
        row, col = np.unravel_index(max_idx, srp_map.shape)
        
        azimuth = theta_grid[row, col]
        elevation = phi_grid[row, col]
    
    return azimuth, elevation, srp_map

def gcc_general(signal1: np.ndarray, 
                signal2: np.ndarray, 
                fs: float, 
                method: str = 'phat') -> Tuple[np.ndarray, np.ndarray]:
    """
    广义互相关(GCC)函数，支持多种加权方案
    
    输入参数：
        signal1, signal2: 输入信号（需等长）
        fs: 采样率 (Hz)
        method: 加权方法，可选：
               'plain' : 普通互相关（无加权）
               'phat'  : PHAT加权（相位变换）
               'scot'  : SCOT加权（平滑相干变换）
               'roth'  : Roth加权
               'ht'    : Hassab-Thompson加权
               'cc'    : 幅度相干加权
    
    输出参数：
        gcc: 广义互相关结果
        lag: 时延轴（秒）
    """
    # 参数检查
    if len(signal1) != len(signal2):
        raise ValueError("输入信号长度必须相同")
    
    n = len(signal1)
    nfft = 2 ** int(np.ceil(np.log2(2 * n - 1)))  # FFT点数
    
    # 计算互功率谱
    F1 = np.fft.fft(signal1, nfft)
    F2 = np.fft.fft(signal2, nfft)
    G12 = F1 * np.conj(F2)  # 互功率谱
    
    # 选择加权函数
    eps_val = 1e-12  # 小值防止除零
    
    if method.lower() == 'plain':
        W = 1.0  # 无加权
    elif method.lower() == 'phat':
        W = 1.0 / (np.abs(G12) + eps_val)  # PHAT加权
    elif method.lower() == 'scot':
        P11 = np.abs(F1) ** 2
        P22 = np.abs(F2) ** 2
        W = 1.0 / np.sqrt(P11 * P22 + eps_val)  # SCOT加权
    elif method.lower() == 'roth':
        P11 = np.abs(F1) ** 2
        W = 1.0 / (P11 + eps_val)  # Roth加权
    elif method.lower() == 'ht':
        P11 = np.abs(F1) ** 2
        P22 = np.abs(F2) ** 2
        W = np.sqrt(1.0 / (P11 * P22 + eps_val))  # Hassab-Thompson加权
    elif method.lower() == 'cc':
        P11 = np.abs(F1) ** 2
        P22 = np.abs(F2) ** 2
        gamma = np.abs(G12) ** 2 / (P11 * P22 + eps_val)  # 相干系数
        W = gamma / (np.abs(G12) * (1 - gamma) + eps_val)  # 幅度相干加权
    else:
        raise ValueError(f"未知加权方法: {method}")
    
    # 应用加权并反变换
    gcc_freq = G12 * W
    gcc = np.fft.ifft(gcc_freq, nfft)
    gcc = np.real(gcc)  # 取实部
    gcc = np.fft.fftshift(gcc)  # 零时延居中
    
    # 生成时延轴（秒）
    lag = np.arange(-nfft//2, nfft//2) / fs
    
    # 确保长度匹配
    if len(lag) > len(gcc):
        lag = lag[:len(gcc)]
    elif len(lag) < len(gcc):
        gcc = gcc[:len(lag)]
    
    return gcc, lag

def process_drone_data(data_values: List[float], 
                              data_order: int = 0, 
                              ArrayNum: int = 4) -> Dict[str, Any]:
    """
    使用TDOA方法处理无人机数据，进行3D定位
    
    Args:
        data_values: 传感器数据数组
        data_order: 数据顺序编号
        ArrayNum: 阵元数量（必须为4，四面体配置）
        
    Returns:
        包含3D定位结果的字典
    """
    try:
        if ArrayNum != 4:
            return {"error": "TDOA方法需要4个阵元（四面体配置）", "mode": "fallback"}
        
        if not data_values:
            return {"error": "空数据数组", "mode": "fallback"}
        
        # 确保所有值都是数字
        try:
            data_values = [float(val) for val in data_values]
        except ValueError as ve:
            return {"error": f"无效数据值: {str(ve)}", "mode": "fallback"}
        
        # 将数据等分成4份，模拟四面体阵列信号
        arrays = np.array_split(data_values, ArrayNum)
        min_length = min(len(arr) for arr in arrays)
        
        # 构建接收信号矩阵 [4, signal_length]
        received_signals = np.zeros((ArrayNum, min_length))
        for i in range(ArrayNum):
            received_signals[i, :] = arrays[i][:min_length]
        
        # 定义四面体麦克风阵列位置 (单位：米)
        # 标准四面体配置
        array_positions = np.array([
            [0.0, 0.8, 0.0],    # 麦克风1
            [-0.7, -0.5, 0.0], # 麦克风2  
            [0.6, -0.6, 0.0],# 麦克风3
            [0.0, 0.0, 1.0]     # 麦克风4
        ]).T  # 转置为 [3, 4]
        
        # 参数设置
        fs = 10000  # 采样率 10kHz
        resolution = 2.0  # 角度分辨率 2度
        
        # 调用TDOA定位算法
        azimuth, elevation, srp_map = estimate_doa_3d_tetrahedral(
            received_signals=received_signals,
            fs=fs,
            array_pos=array_positions,
            resolution=resolution
        )

        # 检查定位是否成功
        if azimuth == 0.0 and elevation == 0.0:
            print("TDOA定位失败：峰值不明显")
            return {
                "error": "定位失败，信号峰值不明显",
                "mode": "fallback",
                "array_num": ArrayNum,
                "signal_length": min_length
            }

        ##绘制SRP能量图!!!!!!!!!#################################################################
        plt.figure(figsize=(10, 8))
        plt.imshow(srp_map, extent=[0, 180, 0, 90], aspect='auto', origin='lower', cmap='jet')
        plt.colorbar(label='SRP energy')
        plt.scatter(azimuth, elevation, color='red', marker='x', s=100, linewidth=2, label='location')
        plt.xlabel('Azimuth angle (deg)')
        plt.ylabel('Elevation angle (deg)')
        plt.title('SRP energy map - 3D source estimate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
                
        # 基于估计角度计算3D坐标
        distance_estimate = 8.0  # 假设距离8米
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)
        
        x = distance_estimate * np.cos(azimuth_rad) * np.cos(elevation_rad)
        y = distance_estimate * np.sin(azimuth_rad) * np.cos(elevation_rad)
        z = distance_estimate * np.sin(elevation_rad)
        
        print(f"TDOA定位结果:")
        print(f"  方位角: {azimuth:.2f}°")
        print(f"  俯仰角: {elevation:.2f}°")
        print(f"  3D坐标: ({x:.3f}, {y:.3f}, {z:.3f})")
        print(f"  估计距离: {distance_estimate:.1f}m")
        
        return {
            "x": round(x, 3),
            "y": round(y, 3),
            "z": round(z, 3),
            "azimuth": round(azimuth, 2),
            "elevation": round(elevation, 2),
            "distance_estimate": round(distance_estimate, 2),
            "array_num": ArrayNum,
            "signal_length": min_length,
            "mode": "tdoa_3d_localization",
            "method": "gcc_phat_tetrahedral"
        }
        
    except Exception as e:
        return {
            "error": f"TDOA处理失败: {str(e)}",
            "mode": "error_fallback"
        }


##################################################################################测试加载函数#####################################################################
def load_input_data():
    """加载输入数据，支持参数和文件两种方式"""
    if len(sys.argv) < 2:
        raise ValueError("Missing arguments")

    arg = sys.argv[1]

    if arg.startswith('--file='):
        # 文件传输模式
        file_path = arg[7:]  # 移除 '--file=' 前缀
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {str(e)}")
    else:
        # 参数传输模式
        try:
            return json.loads(arg)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON argument: {str(e)}")

# 在main函数中加载Excel数据
def load_index_data(excel_file: str = "s_curve_index_table.xlsx") -> Dict:
    """加载索引表数据，只需要执行一次"""
    import pandas as pd
    try:
        df = pd.read_excel(excel_file)
        index_data = {
            'distances': df['距离(m)'].tolist(),
            'x_coords': df['X坐标'].tolist(),
            'y_coords': df['Y坐标'].tolist()
        }
        print(f"成功加载索引表，共 {len(index_data['distances'])} 行数据")
        return index_data
    except Exception as e:
        print(f"加载索引表失败: {e}")
        # 返回空数据，函数会使用默认值
        return {'distances': [], 'x_coords': [], 'y_coords': []}

def load_variance_data_from_excel(filename: str = "variance_data.xlsx") -> List[float]:
    """
    从Excel文件加载方差数据
    """
    try:
        df = pd.read_excel(filename)
        data_values = df['variance'].tolist()
        print(f"成功加载方差数据，长度: {len(data_values)}")
        return data_values
    except Exception as e:
        print(f"加载方差数据失败: {e}")
        return []


"""
主函数
"""
##################################################################################测试加载函数#####################################################################
def main():
    """主函数 - 支持多种输入模式"""
    try:
        data_name = "hallway"
        data_order = 1

        if data_name == "hallway":            # 步态模式
            # 加载步态相关输入数据
            index_data = load_index_data("s_curve_index_table.xlsx")              # 索引表，每个光纤长度，间隔0.1m，对应的走廊的[x,y]坐标
            data_values =  load_variance_data_from_excel("variance_data1.xlsx")    # 方差示例 一列数据，长度为1600

        elif data_name == "single_source":    # 单频定位模式
            data_values =  load_variance_data_from_excel("1KHz_60deg.xlsx")    # 单频时域信号，四个阵列数据拼接成了一列

        elif data_name == "drone":            # 无人机定位模式
            data_values =  load_variance_data_from_excel("drone_data.xlsx")    # 无人机时域信号，四个阵列数据拼接成了一列

        # 验证数据类型
        if not isinstance(data_values, list):
            raise ValueError("dataValues must be an array")

        # 根据数据类型选择处理方法
        if data_name == "hallway":
            result = process_hallway_data(data_values, index_data)
            # 调用函数并打印结果
            print("\n=== 函数执行结果 ===")
            print("返回结果字典:")
            for key, value in result.items():
                print(f"  {key}: {value}")

            # 或者更详细的格式化输出
            print("\n=== 详细坐标信息 ===")
            print(f"前段位置:")
            print(f"  距离: {result['front_distance']}m")
            print(f"  索引: {result['front_index']}")
            print(f"  坐标: ({result['front_x']}, {result['front_y']})")

            print(f"后段位置:")
            print(f"  距离: {result['back_distance']}m")
            print(f"  索引: {result['back_index']}")
            print(f"  坐标: ({result['back_x']}, {result['back_y']})")
            print(f"最终位置:")
            print(f"  坐标: ({result['True_x']}, {result['True_y']})")

        elif data_name == "single_source":
            result = process_single_source_data(data_values, data_order, ArrayNum=4)
                      # 输出结果
            print("\n=== 波束形成单频声源定位结果 ===")
            if "error" in result:
                print(f"处理错误: {result['error']}")
            else:
                print(f"方位角: {result['theta_estimate']}°")
                print(f"距离估计: {result['distance_estimate']}°") 
                print(f"3D坐标: ({result['x']}, {result['y']}, {result['z']})")
                print(f"处理模式: {result['mode']}")
                print(f"使用方法: {result['method']}")

        elif data_name == "drone":
            # 使用TDOA方法
            result = process_drone_data(data_values, data_order, ArrayNum=4)
            
            # 输出结果
            print("\n=== TDOA无人机定位结果 ===")
            if "error" in result:
                print(f"处理错误: {result['error']}")
            else:
                print(f"方位角: {result['azimuth']}°")
                print(f"俯仰角: {result['elevation']}°") 
                print(f"3D坐标: ({result['x']}, {result['y']}, {result['z']})")
                print(f"处理模式: {result['mode']}")
                print(f"使用方法: {result['method']}")
        else:
            result = {"x": 0, "y": 0, "error": f"Unknown data type: {data_name}"}

    except Exception as e:
        error_result = {
            "error": f"Script execution failed: {str(e)}",
            "x": 0,
            "y": 0,
            "processing_mode": "error",
            "timestamp": __import__('time').time()
        }
        print(json.dumps(error_result))


if __name__ == "__main__":
    main()
    