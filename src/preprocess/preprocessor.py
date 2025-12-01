import logging
import pandas as pd
from typing import Optional, List, Tuple
from src.utils.data_utils import (
    filter_and_rename_columns,
    remove_duplicates,
    remove_non_numeric_rows_and_encode_label,
    normalize_data,
    split_train_test,
    remove_inf_nan_rows
)
from src.utils.io import read_csv_safe, save_csv_safe
from src.constant import ( 
    RAW_FEATURES, 
    PROCESSED_FEATURES, 
    RAW_DATA_FILES, 
    NORMALIZATION_METHODS,
    PROCESSED_OUTPUTS_TRAIN,
    PROCESSED_OUTPUTS_TEST
)
from src.utils.logger import get_logger

logger = get_logger("preprocessor")

def preprocess_and_split_csv(
    input_path: str,
    output_train_path: str,
    output_test_path: str,
    raw_features: List[str],
    processed_features: List[str],
    normalization_method: str = 'minmax',
    test_size: float = 0.2,
    stratify_col: str = "label",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Xử lý và chia dữ liệu thành train/test rồi lưu file, trả về DataFrames.
    """
    logger.info(f"Đọc dữ liệu từ: {input_path}")
    df = read_csv_safe(input_path)
    logger.info(f"Data shape ban đầu: {df.shape}")

    df = filter_and_rename_columns(df, raw_features, processed_features)
    logger.info(f"Sau khi lọc/đổi tên feature: {df.shape}")

    df = remove_duplicates(df)
    logger.info(f"Sau khi loại bỏ dòng trùng: {df.shape}")

    df = remove_non_numeric_rows_and_encode_label(df, label_col="label")
    logger.info(f"Sau khi loại hàng non-numeric và encode label: {df.shape}")
    
    df = remove_inf_nan_rows(df, label_col="label")

    logger.info(f"Tiến hành chuẩn hóa kiểu: {normalization_method}")
    df = normalize_data(
        df,
        columns=[col for col in df.columns if col != "label"],
        method=normalization_method
    )
    logger.info(f"Sau chuẩn hóa: {df.shape}")

    # Chia train/test
    df_train, df_test = split_train_test(
        df, test_size=test_size, stratify_col=stratify_col, random_state=random_state
    )
    logger.info(f"Chia train/test: train={df_train.shape}, test={df_test.shape}")

    # Lưu file
    save_csv_safe(df_train, output_train_path)
    logger.info(f"Đã lưu file train: {output_train_path}")
    save_csv_safe(df_test, output_test_path)
    logger.info(f"Đã lưu file test: {output_test_path}")

    return df_train, df_test

if __name__ == "__main__":
    # Xử lý hàng loạt các file trong RAW_DATA_FILES
    from pathlib import Path

    logger.info("=== BẮT ĐẦU TIỀN XỬ LÝ NHIỀU FILE ===")

    for i in range(len(RAW_DATA_FILES)):
        input_path = RAW_DATA_FILES[i]
        output_train = PROCESSED_OUTPUTS_TRAIN[i]
        output_test  = PROCESSED_OUTPUTS_TEST[i]

        logger.info(f"--- Xử lý file: {input_path} ---")
        df_train, df_test = preprocess_and_split_csv(
            input_path,
            output_train,
            output_test,
            raw_features=RAW_FEATURES,
            processed_features=PROCESSED_FEATURES,
            normalization_method=NORMALIZATION_METHODS[0],
            test_size=0.2,
            stratify_col="label",
            random_state=42,
        )
        logger.info(f"File {input_path} hoàn thành! Output: {output_train}, {output_test}")

    logger.info("=== ĐÃ XỬ LÝ XONG TOÀN BỘ FILE ===")
    