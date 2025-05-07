import os
import pandas as pd

# 构造输入输出文件路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "train", "2020-05-14-US-MTV-1", "Pixel4"))
input_file = os.path.join(base_dir, "Pixel4_GnssLog.txt")
output_file = os.path.join(base_dir, "Pixel4_GnssRaw.csv")

# 输出字段（包含转换后的时间）
columns = ["millisSinceGpsEpoch", "svid", "Cn0DbHz", "pseudorangeRateMetersPerSecond", "constellationType"]

# 初始模拟 GPS 时间戳
current_gps_time = 1273529462442
last_unix_time = None

# Raw 字段索引（从 Raw 行标准格式中确定）
field_indices = {
    "utcTimeMillis": 1,
    "Svid": 11,
    "Cn0DbHz": 16,
    "PseudorangeRateMetersPerSecond": 17,
    "ConstellationType": 28
}

filtered_data = []

with open(input_file, "r") as f:
    for line in f:
        if line.startswith("#"):
            continue  # 跳过注释行
        parts = line.strip().split(",")
        if len(parts) <= field_indices["ConstellationType"]:
            continue  # 行长度不足，跳过
        if parts[0].strip() == "Raw":
            try:
                unix_time = int(parts[field_indices["utcTimeMillis"]])
            except ValueError:
                continue  # 非法时间戳，跳过

            # 检查是否为新的时间组
            if unix_time != last_unix_time:
                last_unix_time = unix_time
                current_gps_time += 1000

            # 提取所需字段
            row = [
                current_gps_time,
                parts[field_indices["Svid"]],
                parts[field_indices["Cn0DbHz"]],
                parts[field_indices["PseudorangeRateMetersPerSecond"]],
                parts[field_indices["ConstellationType"]]
            ]
            filtered_data.append(row)

# 保存为 CSV
df = pd.DataFrame(filtered_data, columns=columns)
df.to_csv(output_file, index=False)

print(f"处理完成，已保存至 {output_file}")
