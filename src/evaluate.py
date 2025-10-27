# src/evaluate.py (Phiên bản rút gọn)

import numpy as np
import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
# import folium # Tạm thời không cần
from src import config, models, dataset
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanAbsoluteError
from torch import nn
from src.train import run_epoch  # Import từ src.train
# Import hàm _create_sequences từ data_processing
from src.data_processing import _create_sequences


# --- Hàm hỗ trợ tải model (Không đổi) ---
def _load_model(model_type: str, device: torch.device):
    if model_type == 'pytorch':
        checkpoint_path = config.CKPT_PATH_PYTORCH
        params = config.MODEL_PARAMS['pytorch']
        ModelClass = models.LSTMForecaster
    elif model_type == 'scratch':
        checkpoint_path = config.CKPT_PATH_SCRATCH
        params = config.MODEL_PARAMS['scratch']
        ModelClass = models.LSTMFromScratchForecaster
    else:
        raise ValueError("Unknown model type")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy checkpoint: {checkpoint_path}")
        print(f"Vui lòng huấn luyện model '{model_type}' trước.")
        return None, None
    model = ModelClass(
        in_dim=checkpoint['in_dim'],
        out_dim=checkpoint['out_dim'],
        hidden=params['hidden'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


# --- Hàm vẽ bản đồ (Đã xóa) ---
# (Chúng ta sẽ thêm lại sau khi bạn sẵn sàng)


# --- Hàm chính (Đã cập nhật) ---
def run_evaluation_and_plot():
    """
    Chạy đánh giá chung trên tập test (dự đoán delta)
    """
    print("\n--- Bắt đầu Đánh giá và So sánh Model (Logic Delta) ---")

    # === PHẦN 1: ĐÁNH GIÁ CHUNG TRÊN TẬP TEST (TỪ .NPZ) ===

    try:
        npz = np.load(config.PROCESSED_NPZ_PATH, allow_pickle=True)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {config.PROCESSED_NPZ_PATH}.")
        return

    X_test, y_test = npz["X_test"], npz["y_test"]  # y_test bây giờ là deltas
    INPUT_FEATURES = list(npz["INPUT_FEATURES"])
    TARGET_FEATURES = list(npz["TARGET_FEATURES"])  # Sẽ là ['delta_lat', 'delta_lon']

    test_loader = DataLoader(dataset.StormSeqDataset(X_test, y_test), batch_size=config.BATCH_SIZE, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crit = nn.MSELoss()
    mae_metric = MeanAbsoluteError().to(device)

    model_pytorch, _ = _load_model('pytorch', device)
    model_scratch, _ = _load_model('scratch', device)

    if model_pytorch is None or model_scratch is None:
        print("Thiếu model, không thể so sánh. Dừng lại.")
        return

    print("Đang đánh giá Model PyTorch (nn.LSTM)...")
    test_loss_pt, test_mae_pt = run_epoch(test_loader, model_pytorch, crit, None, device, False, mae_metric)
    print(f"[PyTorch]  Test MSE (delta): {test_loss_pt:.6f} | Test MAE (delta): {test_mae_pt:.6f}")

    print("Đang đánh giá Model From-Scratch (ManualLSTM)...")
    test_loss_sc, test_mae_sc = run_epoch(test_loader, model_scratch, crit, None, device, False, mae_metric)
    print(f"[Scratch]  Test MSE (delta): {test_loss_sc:.6f} | Test MAE (delta): {test_mae_sc:.6f}")

    # Vẽ biểu đồ thanh (Cập nhật nhãn)
    print(f"\nĐang vẽ biểu đồ so sánh cho mẫu 0 (từ .npz)...")

    # y_true bây giờ là delta
    y_true_sample_for_bar = y_test[0].flatten()
    x_sample_for_bar = torch.tensor(X_test[0], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred_pt_sample = model_pytorch(x_sample_for_bar).cpu().numpy().flatten()
        y_pred_sc_sample = model_scratch(x_sample_for_bar).cpu().numpy().flatten()

    labels = TARGET_FEATURES  # ['delta_lat', 'delta_lon']
    x_pos = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x_pos - width, y_true_sample_for_bar, width, label='Ground Truth (Delta)', color='navy')
    rects2 = ax.bar(x_pos, y_pred_pt_sample, width, label='PyTorch LSTM (Delta)', color='red')
    rects3 = ax.bar(x_pos + width, y_pred_sc_sample, width, label='Scratch LSTM (Delta)', color='orange')

    ax.set_ylabel('Giá trị (Scaled Deltas)')  # <-- Đã cập nhật
    ax.set_title(f'So sánh dự đoán Delta (1 bước) cho Mẫu Test 0')  # <-- Đã cập nhật

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    config.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.COMPARISON_PLOT_PATH)
    print(f"Đã lưu biểu đồ so sánh tại: {config.COMPARISON_PLOT_PATH}")

    print("--- Hoàn tất Đánh giá (Logic Delta) ---")
    print("\nPhần vẽ bản đồ tự hồi quy đã được tạm thời vô hiệu hóa.")
    print("Sau khi huấn luyện xong, hãy cho tôi biết để tôi cung cấp logic vẽ bản đồ mới.")