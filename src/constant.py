
from typing import List

# # Danh sách các cột features sử dụng trong các tác vụ phân tích/học máy
# RAW_FEATURES: List[str] = [
#     "Flow Duration",
#     "Total Fwd Packets",
#     "Total Bwd Packets",
#     "Total Length of Fwd Packets",
#     "Total Length of Bwd Packets",
#     "Fwd Packet Length Mean",
#     "Fwd Packet Length Std",
#     "Bwd Packet Length Mean",
#     "Bwd Packet Length Std",
#     "Flow Bytes/s",
#     "Flow Packets/s",
#     "Flow IAT Mean",
#     "Flow IAT Std",
#     "Fwd IAT Mean",
#     "Fwd IAT Std",
#     "Bwd IAT Mean",
#     "Bwd IAT Std",
#     "Active Mean",
#     "Active Std",
#     "Idle Mean",
#     "Idle Std",
#     "Protocol",
#     "FIN Flag Count",
#     "SYN Flag Count",
#     "RST Flag Count",
#     "ACK Flag Count",
#     "Down/Up Ratio",
#     "Init_Win_bytes_forward",
#     "Init_Win_bytes_backward",
#     "Fwd Seg Size Avg",
#     "Bwd Seg Size Avg",
#     "BytesPerFwdPacket",
#     "BytesPerBwdPacket",
#     "FwdBwdPacketRatio",
#     "Label"
# ]

# PROCESSED_FEATURES: List[str] = [
#     "flow_duration",
#     "total_fwd_packets",
#     "total_bwd_packets",
#     "total_length_of_fwd_packets",
#     "total_length_of_bwd_packets",
#     "fwd_packet_length_mean",
#     "fwd_packet_length_std",
#     "bwd_packet_length_mean",
#     "bwd_packet_length_std",
#     "flow_bytes_per_second",
#     "flow_packets_per_second",
#     "flow_iat_mean",
#     "flow_iat_std",
#     "fwd_iat_mean",
#     "fwd_iat_std",
#     "bwd_iat_mean",
#     "bwd_iat_std",
#     "active_mean",
#     "active_std",
#     "idle_mean",
#     "idle_std",
#     "protocol",
#     "fin_flag_count",
#     "syn_flag_count",
#     "rst_flag_count",
#     "ack_flag_count",
#     "down_up_ratio",
#     "init_win_bytes_forward",
#     "init_win_bytes_backward",
#     "fwd_seg_size_avg",
#     "bwd_seg_size_avg",
#     "bytes_per_fwd_packet",
#     "bytes_per_bwd_packet",
#     "fwd_bwd_packet_ratio",
#     "label"
# ]

RAW_FEATURES: List[str] = [
    # Thống kê tổng và chiều gói/byte
    'Flow Duration',
    'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',

    # Thống kê độ dài/tốc độ
    'Flow Bytes/s', 'Flow Packets/s',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'Min Packet Length', 'Max Packet Length',

    # Các flag và chỉ số trạng thái
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',

    # Forward/backward packet thống kê
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean','Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean','Bwd Packet Length Std',
    'Fwd Packets/s', 'Bwd Packets/s',
    'Down/Up Ratio', 'Average Packet Size',

    # Độ trễ và các IAT đặc trưng
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',

    # Các chỉ số về các cờ đặc biệt
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',

    # Thông tin active/idle của flow
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',

    # Các trường nâng cao/bulk segment, window
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 
    'act_data_pkt_fwd', 'min_seg_size_forward',

    # Nhãn mục tiêu
    'Label'
]

PROCESSED_FEATURES: List[str] = [
    'flow_duration',
    'total_fwd_packets', 'total_backward_packets',
    'total_length_of_fwd_packets', 'total_length_of_bwd_packets',
    'flow_bytes_per_second', 'flow_packets_per_second',
    'packet_length_mean', 'packet_length_std', 'packet_length_variance',
    'min_packet_length', 'max_packet_length',
    'fin_flag_count', 'syn_flag_count', 'rst_flag_count', 'psh_flag_count',
    'ack_flag_count', 'urg_flag_count', 'cwe_flag_count', 'ece_flag_count',
    'fwd_packet_length_max', 'fwd_packet_length_min', 'fwd_packet_length_mean','fwd_packet_length_std',
    'bwd_packet_length_max', 'bwd_packet_length_min', 'bwd_packet_length_mean','bwd_packet_length_std',
    'fwd_packets_per_second', 'bwd_packets_per_second',
    'down_up_ratio', 'average_packet_size',
    'flow_iat_mean', 'flow_iat_std', 'flow_iat_max',
    'flow_iat_min',
    'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min',
    'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min',
    'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags',
    'active_mean', 'active_std', 'active_max', 'active_min',    
    'idle_mean', 'idle_std', 'idle_max', 'idle_min',
    'init_win_bytes_forward', 'init_win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward',
    'label'
]

RAW_DATA_FILES: List[str] = [
    "D:/DOCUMENTS/SEMESTER/20251/Đồ án tốt nghiệp/CODE/data/raw/cicddos2019/drdos_dns.csv",
    "D:/DOCUMENTS/SEMESTER/20251/Đồ án tốt nghiệp/CODE/data/raw/cicddos2019/drdos_udp.csv",
]

PROCESSED_OUTPUTS_TRAIN: List[str] = [
    "D:/DOCUMENTS/SEMESTER/20251/Đồ án tốt nghiệp/CODE/data/processed/train/cicddos2019/drdos_dns_training.csv",
    "D:/DOCUMENTS/SEMESTER/20251/Đồ án tốt nghiệp/CODE/data/processed/train/cicddos2019/drdos_udp_training.csv"
]

PROCESSED_OUTPUTS_TEST: List[str] = [
    "D:/DOCUMENTS/SEMESTER/20251/Đồ án tốt nghiệp/CODE/data/processed/test/cicddos2019/drdos_dns_testing.csv",
    "D:/DOCUMENTS/SEMESTER/20251/Đồ án tốt nghiệp/CODE/data/processed/test/cicddos2019/drdos_udp_testing.csv"
]

NORMALIZATION_METHODS: List[str] = [
    "minmax",       # Min-Max Scaling: Đưa dữ liệu về [0, 1]
    "zscore",       # Standardization (Z-score): Đưa về mean=0, std=1
    "maxabs",       # MaxAbsScaler: Đưa dữ liệu về [-1, 1] dựa vào giá trị tuyệt đối lớn nhất
    "robust",       # RobustScaler: Ít nhạy cảm với outlier (chuẩn hóa theo median và IQR)
    "l1",           # L1 Normalization: Vector norm = 1
    "l2",           # L2 Normalization: (chuẩn hóa vector về norm 2 = 1)
]