import pandas as pd
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, Normalizer
)


def filter_and_rename_columns(
    df: pd.DataFrame,
    raw_features: List[str],
    processed_features: List[str]
) -> pd.DataFrame:

    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    key_features = [raw.strip() for raw in raw_features]
    mapping = dict(zip(key_features, processed_features))
    rename_map = {col: mapping[col] for col in df.columns if col in mapping}
    df = df.rename(columns=rename_map)
    keep_cols = [col for col in processed_features if col in df.columns]
    return df[keep_cols]


import pandas as pd
from typing import List, Optional

def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> pd.DataFrame:
    cleaned_df = df.drop_duplicates(
        subset=subset,
        keep=keep,
        ignore_index=True
    )
    return cleaned_df

import pandas as pd
from typing import List

def remove_non_numeric_rows_and_encode_label(
    df: pd.DataFrame,
    label_col: str = "label"
) -> pd.DataFrame:
    cols_to_check = [col for col in df.columns if col != label_col]
    filtered_df = df.copy()
    filtered_df[cols_to_check] = filtered_df[cols_to_check].apply(
        pd.to_numeric, errors='coerce'
    )
    filtered_df = filtered_df.dropna(subset=cols_to_check).reset_index(drop=True)
    filtered_df[label_col] = filtered_df[label_col].apply(
        lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1
    )
    return filtered_df

def normalize_data(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'minmax'
) -> pd.DataFrame:
    df_norm = df.copy()
    if columns is None:
        columns = df_norm.select_dtypes(include='number').columns.tolist()

    scaler_map = {
        'minmax': MinMaxScaler(),
        'zscore': StandardScaler(),
        'maxabs': MaxAbsScaler(),
        'robust': RobustScaler(),
        'l1': Normalizer(norm='l1'),
        'l2': Normalizer(norm='l2'),
    }

    if method not in scaler_map:
        raise ValueError(f"Phương pháp '{method}' không hỗ trợ.")

    scaler = scaler_map[method]

    # Với l1/l2 (chuẩn hóa theo vector hàng), phải xử lý riêng
    if method in ['l1', 'l2']:
        normed_array = scaler.fit_transform(df_norm[columns])
        df_norm[columns] = normed_array
    else:
        df_norm[columns] = scaler.fit_transform(df_norm[columns])

    return df_norm

def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    stratify_col: Optional[str] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chia DataFrame thành tập train và test.

    Args:
        df: DataFrame đầu vào.
        test_size: Tỷ lệ mẫu thuộc tập test (ví dụ 0.2 là 20% cho test).
        stratify_col: Tên cột dùng để stratify (nên là tên nhãn nếu là phân loại).
        random_state: Seed để chia dữ liệu có thể lặp lại.

    Returns:
        (df_train, df_test): Tuple DataFrame train & test.
    """
    stratify_vals = df[stratify_col] if stratify_col else None
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_vals
    )
    # reset chỉ số dòng (index)
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

def remove_inf_nan_rows(
    df: pd.DataFrame,
    label_col: str = "label"
) -> pd.DataFrame:
    """
    Loại bỏ hàng có giá trị NaN, inf, -inf ở các cột đặc trưng (trừ cột label).
    """
    cols_to_check = [col for col in df.columns if col != label_col]
    df = df.replace([float('inf'), float('-inf')], pd.NA)
    df = df.dropna(subset=cols_to_check).reset_index(drop=True)
    return df