# main.py (Cập nhật)
import argparse
import sys
from pathlib import Path

# Thêm src vào path để import
sys.path.append(str(Path(__file__).resolve().parent))

# Import thêm hàm mới
from src import prepare_raw_data, data_processing, train, evaluate


def main():
    parser = argparse.ArgumentParser(description="Pipeline dự báo quỹ đạo bão.")

    # Thêm Argument mới cho Bước 1
    parser.add_argument(
        '--prepare-raw-data',
        action='store_true',
        help="Bước 1: Cắt file CSV lớn (ibtracs.last3years...) thành ibtracs_track_ml.csv."
    )

    parser.add_argument(
        '--process-data',
        action='store_true',
        help="Bước 2: Tiền xử lý dữ liệu (ibtracs_track_ml.csv -> NPZ)."
    )

    parser.add_argument(
        '--train',
        nargs='+',  # Chấp nhận một hoặc nhiều giá trị
        choices=['pytorch', 'scratch', 'all'],
        help="Bước 3: Huấn luyện model. Chọn 'pytorch', 'scratch', hoặc 'all'."
    )

    parser.add_argument(
        '--evaluate',
        action='store_true',
        help="Bước 4: Chạy đánh giá, so sánh 2 model và vẽ biểu đồ."
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help="Chạy tất cả các bước: 1, 2, 3 (all models), và 4."
    )

    args = parser.parse_args()

    # Xử lý --all (Thêm bước 1)
    if args.all:
        args.prepare_raw_data = True
        args.process_data = True
        args.train = ['all']
        args.evaluate = True

    # --- Thực thi các bước theo thứ tự ---

    # Bước 1
    if args.prepare_raw_data:
        prepare_raw_data.run_raw_data_preparation()

    # Bước 2
    if args.process_data:
        data_processing.run_data_pipeline()

    # Bước 3
    if args.train:
        if 'all' in args.train:
            train.run_training(model_type='pytorch')
            train.run_training(model_type='scratch')
        else:
            if 'pytorch' in args.train:
                train.run_training(model_type='pytorch')
            if 'scratch' in args.train:
                train.run_training(model_type='scratch')

    # Bước 4
    if args.evaluate:
        evaluate.run_evaluation_and_plot()

    # In trợ giúp nếu không có đối số nào được cung cấp
    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()