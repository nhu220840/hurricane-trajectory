# src/evaluate.py
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import folium
from src import config, models, dataset
from src.train import run_epoch
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanAbsoluteError
from torch import nn


# --- Hàm hỗ trợ tải model ---
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
        checkpoint = torch.load(checkpoint_path, map_location=device)
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


# --- Hàm hỗ trợ vẽ bản đồ (từ visualize_folium.py cũ) ---
def _reconstruct_path(start_point, deltas):
    """Tính toán đường đi từ điểm bắt đầu và các delta dự đoán."""
    # (Bạn có thể cần điều chỉnh logic này nếu target của bạn là giá trị tuyệt đối)
    # Giả sử target là 'lat', 'lon' tuyệt đối
    return deltas


def _draw_map(history_coords, truth_path, pred_path, out_html, model_name):
    start_point = history_coords[-1]
    m = folium.Map(location=start_point, zoom_start=7, tiles="CartoDB positron")

    folium.PolyLine(history_coords, color="gray", weight=3, tooltip="Lịch sử").add_to(m)
    folium.PolyLine([start_point] + truth_path, color="navy", weight=5, tooltip=f"Thực tế").add_to(m)
    folium.PolyLine([start_point] + pred_path, color="red", weight=3, dash_array="10, 5",
                    tooltip=f"Dự đoán ({model_name})").add_to(m)

    m.save(out_html)
    print(f"Đã lưu bản đồ so sánh tại: '{out_html}'")


# --- Hàm chính: Đánh giá và So sánh ---

def run_evaluation_and_plot(sample_index: int = 0):
    """
    Chạy đánh giá trên tập test cho CẢ HAI model và vẽ biểu đồ so sánh.
    """
    print("\n--- Bắt đầu Đánh giá và So sánh Model ---")

    # 1. Tải dữ liệu Test và Scaler
    try:
        npz = np.load(config.PROCESSED_NPZ_PATH, allow_pickle=True)
        with open(config.SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {config.PROCESSED_NPZ_PATH} hoặc {config.SCALER_PATH}.")
        print("Vui lòng chạy bước '--process-data' trước.")
        return

    X_test, y_test = npz["X_test"], npz["y_test"]
    INPUT_FEATURES = list(npz["INPUT_FEATURES"])
    TARGET_FEATURES = list(npz["TARGET_FEATURES"])

    test_loader = DataLoader(dataset.StormSeqDataset(X_test, y_test), batch_size=config.BATCH_SIZE, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crit = nn.MSELoss()
    mae_metric = MeanAbsoluteError().to(device)

    # 2. Tải cả hai model
    model_pytorch, _ = _load_model('pytorch', device)
    model_scratch, _ = _load_model('scratch', device)

    if model_pytorch is None or model_scratch is None:
        print("Thiếu model, không thể so sánh. Dừng lại.")
        return

    # 3. Đánh giá toàn bộ tập Test
    print("Đang đánh giá Model PyTorch (nn.LSTM)...")
    test_loss_pt, test_mae_pt = run_epoch(test_loader, model_pytorch, crit, None, device, False, mae_metric)
    print(f"[PyTorch]  Test MSE: {test_loss_pt:.6f} | Test MAE: {test_mae_pt:.6f}")

    print("Đang đánh giá Model From-Scratch (ManualLSTM)...")
    test_loss_sc, test_mae_sc = run_epoch(test_loader, model_scratch, crit, None, device, False, mae_metric)
    print(f"[Scratch]  Test MSE: {test_loss_sc:.6f} | Test MAE: {test_mae_sc:.6f}")

    # 4. Lấy dự đoán cho một mẫu cụ thể (sample_index)
    x_sample = torch.tensor(X_test[sample_index], dtype=torch.float32).unsqueeze(0).to(device)
    y_true_sample = y_test[sample_index]  # Shape [1, 2]

    with torch.no_grad():
        y_pred_pt_sample = model_pytorch(x_sample).cpu().numpy().flatten()
        y_pred_sc_sample = model_scratch(x_sample).cpu().numpy().flatten()

    # (Giả sử y_true, y_pred đều là [lat, lon])
    y_true = y_true_sample.flatten()

    # 5. Vẽ Biểu đồ Matplotlib (Yêu cầu chính của bạn)
    print(f"Đang vẽ biểu đồ so sánh cho mẫu {sample_index}...")

    labels = TARGET_FEATURES  # ['lat', 'lon']
    x_pos = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x_pos - width, y_true, width, label='Ground Truth', color='navy')
    rects2 = ax.bar(x_pos, y_pred_pt_sample, width, label='PyTorch LSTM', color='red')
    rects3 = ax.bar(x_pos + width, y_pred_sc_sample, width, label='Scratch LSTM', color='orange')

    ax.set_ylabel('Giá trị đã chuẩn hóa (Scaled)')
    ax.set_title(f'So sánh dự đoán cho Mẫu Test {sample_index}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    config.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.COMPARISON_PLOT_PATH)
    print(f"Đã lưu biểu đồ so sánh tại: {config.COMPARISON_PLOT_PATH}")

    # 6. (Tùy chọn) Vẽ bản đồ Folium cho cả hai
    # Bạn sẽ cần logic để giải nén (inverse_transform) tọa độ
    # (Phần này phức tạp hơn, tùy thuộc vào dữ liệu của bạn là delta hay tuyệt đối)
    # ... (Bỏ qua phần vẽ map phức tạp để tập trung vào biểu đồ) ...

    print("--- Hoàn tất Đánh giá ---")