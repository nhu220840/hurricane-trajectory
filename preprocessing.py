import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

# --- Bước 1: Đọc và Chuẩn bị Dữ liệu (Giống như trước) ---
try:
    df = pd.read_csv('data/ibtracs_track_ml.csv', keep_default_na=False, na_values=[' '])
except FileNotFoundError:
    print("LỖI: Không tìm thấy file 'ibtracs_track_ml.csv'.")
    exit()

# Lọc và đổi tên các cột
columns_to_keep = {
    'SID': 'sid', 'ISO_TIME': 'time', 'LAT': 'lat', 'LON': 'lon',
    'WMO_WIND': 'wind', 'WMO_PRES': 'pres', 'STORM_SPEED': 'speed', 'STORM_DIR': 'direction'
}
df_clean = df[list(columns_to_keep.keys())].rename(columns=columns_to_keep)

# Chuyển đổi kiểu dữ liệu và sắp xếp
df_clean['time'] = pd.to_datetime(df_clean['time'])
feature_cols = ['lat', 'lon', 'wind', 'pres', 'speed', 'direction']
for col in feature_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.sort_values(by=['sid', 'time'])


# --- Bước 2: TẠO CUSTOM TRANSFORMER ĐÃ CẢI TIẾN ---

class GroupedInterpolator(BaseEstimator, TransformerMixin):
    def __init__(self, group_col='sid', feature_cols=None, method='linear'):
        self.group_col = group_col
        self.feature_cols = feature_cols
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Tạo một bản sao để tránh thay đổi DataFrame gốc
        X_copy = X.copy()

        # *** THAY ĐỔI QUAN TRỌNG: CHỈ NỘI SUY TRÊN CÁC CỘT SỐ ***
        interpolated_features = X_copy.groupby(self.group_col)[self.feature_cols].transform(
            lambda x: x.interpolate(method=self.method)
        )

        # Cập nhật lại các cột đã nội suy vào DataFrame
        X_copy[self.feature_cols] = interpolated_features

        # Vẫn điền các giá trị NaN còn sót lại ở đầu/cuối chuỗi
        X_filled = X_copy.groupby(self.group_col, group_keys=False).apply(
            lambda group: group.fillna(method='bfill').fillna(method='ffill')
        )
        # Xóa các hàng vẫn còn NaN (nếu có)
        X_filled.dropna(inplace=True)

        return X_filled


# --- Bước 3: Định nghĩa và Áp dụng Pipeline ---

numeric_features = ['lat', 'lon', 'wind', 'pres', 'speed', 'direction']

# Tạo pipeline tổng thể
full_pipeline = Pipeline(steps=[
    # Bước 1: Sử dụng GroupedInterpolator đã cải tiến
    ('custom_interpolator', GroupedInterpolator(group_col='sid', feature_cols=numeric_features)),

    # Bước 2: Áp dụng scaler cho các cột số
    ('preprocessor', ColumnTransformer(transformers=[
        ('scaler', MinMaxScaler(), numeric_features)
    ], remainder='passthrough'))
])

# Áp dụng pipeline vào dữ liệu
data_transformed = full_pipeline.fit_transform(df_clean)

# Tạo lại DataFrame từ kết quả
new_column_names = numeric_features + [col for col in df_clean.columns if col not in numeric_features]
df_final = pd.DataFrame(data_transformed, columns=new_column_names)

# Sắp xếp lại thứ tự cột
df_final = df_final[['sid', 'time'] + numeric_features]

# --- KẾT QUẢ ---
print("\n--- TIỀN XỬ LÝ HOÀN TẤT (ĐÃ FIX WARNING) ---")
print("5 dòng đầu tiên của dữ liệu đã được xử lý:")
print(df_final.head().to_string())

# Lưu file đã xử lý
output_filename = 'ibtracs_processed_interpolated_pipeline_v2.csv'
df_final.to_csv(output_filename, index=False)
print(f"\nĐã lưu dữ liệu sạch vào file '{output_filename}'.")