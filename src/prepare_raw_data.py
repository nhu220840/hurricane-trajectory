# src/prepare_raw_data.py
import pandas as pd
from src import config

# Đây là các cột mà pipeline ở bước 2 (data_processing.py) cần
# Dựa trên file 'ibtracs.last3years.list.v04r01.csv'
COLUMNS_TO_KEEP = [
    'SID',
    'ISO_TIME',
    'LAT',
    'LON',
    'WMO_WIND',
    'WMO_PRES',
    'STORM_SPEED',
    'STORM_DIR'
]

def run_raw_data_preparation():
    """
    Đọc file CSV thô (lớn) và cắt/lọc ra các cột cần thiết,
    lưu thành 'ibtracs_track_ml.csv' để các bước sau sử dụng.
    """
    print("--- Bắt đầu Bước 1: Chuẩn bị file CSV thô ---")

    # Đảm bảo thư mục 'data/raw' tồn tại
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Đang đọc file CSV lớn từ: {config.RAW_IBTRACS_FILE}")

        # Đọc file CSV lớn
        # Sử dụng các tham số giống như file preprocessing.py cũ để đảm bảo
        # xử lý đúng các giá trị rỗng/khoảng trắng
        df = pd.read_csv(
            config.RAW_IBTRACS_FILE,
            keep_default_na=False,
            na_values=[' '],
            low_memory=False  # Thêm vào để tối ưu khi đọc file lớn
        )
        print(f"Đã đọc xong file. Tổng số dòng: {len(df)}")

    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file data thô: {config.RAW_IBTRACS_FILE}")
        print(f"Vui lòng đảm bảo file 'ibtracs.last3years.list.v04r01.csv' nằm trong thư mục '{config.RAW_DATA_DIR}'.")
        return
    except Exception as e:
        print(f"LỖI khi đọc file: {e}")
        return

    # Cắt lấy các cột cần thiết
    try:
        df_cut = df[COLUMNS_TO_KEEP]
    except KeyError as e:
        print(f"LỖI: Thiếu cột trong file CSV thô. Không tìm thấy cột: {e}")
        print(f"Các cột có sẵn: {df.columns.to_list()}")
        return

    # Lưu file đã cắt (đây sẽ là input cho bước 2)
    try:
        df_cut.to_csv(config.RAW_DATA_PATH, index=False)
        print(f"Đã lưu file đã cắt ({len(df_cut)} dòng) vào: {config.RAW_DATA_PATH}")
    except Exception as e:
        print(f"LỖI khi lưu file: {e}")
        return

    print("--- Hoàn tất Bước 1 ---")


if __name__ == "__main__":
    # Cho phép chạy file này độc lập để test
    run_raw_data_preparation()