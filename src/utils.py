import random
import numpy as np
import torch
import os


def set_seed(seed_value=42):
    """Cố định các hạt giống ngẫu nhiên để đảm bảo tính tái lập."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if multi-GPU
        # Các cài đặt này có thể làm chậm quá trình huấn luyện nhưng cần thiết cho tính tái lập hoàn toàn trên GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Đã cố định hạt giống ngẫu nhiên thành: {seed_value}")
