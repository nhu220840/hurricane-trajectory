# main.py (wrapper ở thư mục gốc repo)

import argparse
import sys
from pathlib import Path

# Bảo đảm import được gói src/ khi chạy từ thư mục gốc
REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

# Import các module trong package src
from src import prepare_raw_data, data_processing, train, evaluate

# (Tuỳ chọn) cố định seed ở mức "wrapper"
try:
    from src.utils import set_seed  # nếu bạn có utils.py
except Exception:
    import os, random, numpy as np, torch

    def set_seed(seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def main():
    # Cố định seed sớm (không bắt buộc, nhưng tốt)
    set_seed(42)

    parser = argparse.ArgumentParser(description="Hurricane trajectory pipeline (delta mode)")

    parser.add_argument(
        "--prepare-raw-data",
        action="store_true",
        help="Bước 1: cắt/đổi tên cột từ IBTrACS gốc -> data/raw/ibtracs_track_ml.csv",
    )
    parser.add_argument(
        "--process-data",
        action="store_true",
        help="Bước 2: tiền xử lý -> data/processed/processed_data.npz (+preprocessor/scaler)",
    )
    parser.add_argument(
        "--train",
        nargs="+",
        choices=["pytorch", "scratch", "all"],
        help="Bước 3: huấn luyện: 'pytorch', 'scratch', hoặc 'all'. Có thể truyền nhiều: --train pytorch scratch",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Bước 4: đánh giá checkpoint đã train (in MAE/MSE theo km & theo độ)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Chạy lần lượt 1→4 (prepare, process, train all, evaluate).",
    )

    args = parser.parse_args()

    # Nếu dùng --all thì bật hết các bước
    if args.all:
        args.prepare_raw_data = True
        args.process_data = True
        args.train = ["all"]
        args.evaluate = True

    ran_any = False

    # ---- Bước 1: chuẩn bị dữ liệu thô gọn ----
    if args.prepare_raw_data:
        ran_any = True
        prepare_raw_data.run_raw_data_preparation()

    # ---- Bước 2: xử lý & tạo NPZ ----
    if args.process_data:
        ran_any = True
        # TÊN HÀM CHUẨN TRONG src/data_processing.py
        data_processing.process_and_save_npz()

    # ---- Bước 3: huấn luyện ----
    if args.train:
        ran_any = True
        # TÊN HÀM CHUẨN TRONG src/train.py: train(model_choice)
        # - 'all' => train cả 2
        # - 'pytorch' hoặc 'scratch' => train từng cái
        if "all" in args.train:
            train.train("all")
        else:
            if "pytorch" in args.train:
                train.train("pytorch")
            if "scratch" in args.train:
                train.train("scratch")

    # ---- Bước 4: đánh giá ----
    if args.evaluate:
        ran_any = True
        # TÊN HÀM CHUẨN TRONG src/evaluate.py
        results = evaluate.evaluate()
        if results is not None:
            print("[Results]", results)

    # Nếu không truyền gì, in help
    if not ran_any:
        parser.print_help()


if __name__ == "__main__":
    main()
