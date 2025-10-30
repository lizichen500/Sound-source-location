# 步态识别程序
**data_processor.py**为主程序，其中分为以下三种定位方式：**步态定位，单频声源定位，无人机声源定位**

## 1. 步态识别定位

1.提取方差数据中的217m-970m位置处的数据
2.进行峰值查询：find_peak_with_threshold函数
3.索引表中进行峰值的键值对的索引，得到实际的x，y坐标

## 2. 单频声源定位

**music_calc_2d_far函数**，采用的线性等间距阵列
采用music（DOA）算法实现对方位角进行定位

## 3. 无人机声援定位

**estimate_doa_3d_tetrahedral函数**，采用基于SRP-PHAT算法，根据到达时间差（TDOA）估算每对阵列的时延差，实现对无人机声源的定位


## 4. requirements.txt 
包含该代码所依赖的**第三方库文件**
```bash
pip install requirements.txt
```

