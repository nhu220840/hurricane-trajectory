# main.py
import argparse

# Import các hàm chạy pipeline từ src
from src.data_processing import run_processing
from src.train import run_training
from src.forecast import run_forecasting

def main():
    parser = argparse.ArgumentParser(description="Pipeline dự báo quỹ đạo bão.")
    parser.add_argument(
        '--process-data',
        action='store_true',
        help="Chạy pipeline tiền xử lý dữ liệu."
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help="Chạy pipeline huấn luyện model."
    )
    parser.add_argument(
        '--forecast',
        action='store_true',
        help="Chạy pipeline dự báo và tạo bản đồ."
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help="Chạy tất cả các bước: process-data, train, và forecast."
    )

    args = parser.parse_args()

    if args.all:
        run_processing()
        run_training()
        run_forecasting()
    else:
        if args.process_data:
            run_processing()
        if args.train:
            run_training()
        if args.forecast:
            run_forecasting()

    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()