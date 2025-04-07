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

# GPS 时间起点相对 Unix 时间的偏移量（毫秒）
GPS_UNIX_OFFSET_MS = 315964800000

# 读取并筛选数据
filtered_data = []

with open(input_file, "r") as f:
    for line in f:
        if line.startswith("#"):
            continue  # 跳过表头行
        parts = line.strip().split(",")
        if parts[0] == "Status":  # 仅保留 Status 关键字行
            # 转换 UnixTimeMillis -> millisSinceGpsEpoch
            unix_time = int(parts[1])
            millis_since_gps_epoch = unix_time - GPS_UNIX_OFFSET_MS
            # 修改最后三位为 442
            millis_since_gps_epoch = (millis_since_gps_epoch // 1000) * 1000 + 442
            parts[1] = millis_since_gps_epoch  # 更新时间值
            filtered_data.append(parts)

# 转换为 DataFrame 并保存
df = pd.DataFrame(filtered_data, columns=columns)
df.to_csv(output_file, index=False)

print(f"处理完成，已保存至 {output_file}")
