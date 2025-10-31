# main.py (wrapper in the repo root)

import argparse
import sys
from pathlib import Path

# Ensure the src/ package can be imported when run from the root directory
REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

# Import modules from the src package
from src import prepare_raw_data, data_processing, train, evaluate

# (Optional) set seed at the "wrapper" level
try:
    from src.utils import set_seed  # if you have utils.py
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
    # Set seed early (not mandatory, but good practice)
    set_seed(42)

    parser = argparse.ArgumentParser(description="Hurricane trajectory pipeline (delta mode)")

    parser.add_argument(
        "--prepare-raw-data",
        action="store_true",
        help="Step 1: cut/rename columns from original IBTrACS -> data/raw/ibtracs_track_ml.csv",
    )
    parser.add_argument(
        "--process-data",
        action="store_true",
        help="Step 2: preprocess -> data/processed/processed_data.npz (+preprocessor/scaler)",
    )
    parser.add_argument(
        "--train",
        nargs="+",
        choices=["pytorch", "scratch", "all"],
        help="Step 3: train: 'pytorch', 'scratch', or 'all'. Can pass multiple: --train pytorch scratch",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Step 4: evaluate trained checkpoint (prints MAE/MSE in km & degrees)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run steps 1→4 sequentially (prepare, process, train all, evaluate).",
    )

    args = parser.parse_args()

    # If --all is used, enable all steps
    if args.all:
        args.prepare_raw_data = True
        args.process_data = True
        args.train = ["all"]
        args.evaluate = True

    ran_any = False

    # ---- Step 1: Prepare concise raw data ----
    if args.prepare_raw_data:
        ran_any = True
        prepare_raw_data.run_raw_data_preparation()

    # ---- Step 2: Process & create NPZ ----
    if args.process_data:
        ran_any = True
        # STANDARD FUNCTION NAME IN src/data_processing.py
        data_processing.process_and_save_npz()

    # ---- Step 3: Training ----
    if args.train:
        ran_any = True
        # STANDARD FUNCTION NAME IN src/train.py: train(model_choice)
        # - 'all' => train both
        # - 'pytorch' or 'scratch' => train individually
        if "all" in args.train:
            train.train("all")
        else:
            if "pytorch" in args.train:
                train.train("pytorch")
            if "scratch" in args.train:
                train.train("scratch")

    # ---- Step 4: Evaluation ----
    if args.evaluate:
        ran_any = True
        # STANDARD FUNCTION NAME IN src/evaluate.py
        results = evaluate.evaluate()
        if results is not None:
            print("[Results]", results)

    # If no arguments are given, print help
    if not ran_any:
        parser.print_help()


if __name__ == "__main__":
    main()