import os
import pandas as pd

# 获取主文件夹的路径（脚本所在的 py_scripts 目录的上一级）
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "train", "2020-05-14-US-MTV-1", "Pixel4"))
input_file = os.path.join(base_dir, "Pixel4_GnssLog.txt")
output_file = os.path.join(base_dir, "Pixel4_GnssLog.csv")

# 目标表头
columns = ["Status", "millisSinceGpsEpoch", "SignalCount", "SignalIndex", 
           "ConstellationType", "Svid", "CarrierFrequencyHz", "Cn0DbHz", 
           "AzimuthDegrees", "ElevationDegrees", "UsedInFix", 
           "HasAlmanacData", "HasEphemerisData"]

current_gps_time = 1273529462442
last_unix_time = None  # 上一组的 UnixTimeMillis

# 读取并筛选数据
filtered_data = []

with open(input_file, "r") as f:
    for line in f:
        if line.startswith("#"):
            continue  # 跳过表头行
        parts = line.strip().split(",")
        if parts[0] == "Status":
            unix_time = int(parts[1])
            
            # 判断是否是新的时间组（UnixTimeMillis变化了）
            if unix_time != last_unix_time:
                last_unix_time = unix_time
                current_gps_time += 1000  # 增加时间戳（每组增加1）

            parts[1] = current_gps_time  # 使用当前的时间戳
            filtered_data.append(parts)

# 转换为 DataFrame 并保存
df = pd.DataFrame(filtered_data, columns=columns)
df.to_csv(output_file, index=False)

print(f"处理完成，已保存至 {output_file}")
