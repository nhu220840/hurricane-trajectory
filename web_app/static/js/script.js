// static/js/script.js

// Biến toàn cục để giữ bản đồ
let map = null;

// Hàm để vẽ bản đồ (sẽ được gọi khi có dữ liệu)
function drawMap(data) {
    // Nếu bản đồ chưa được tạo, hãy tạo nó
    if (!map) {
        map = L.map('map').setView(data.start_point, 7); // Phóng to vào điểm bắt đầu
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
    } else {
        // Nếu bản đồ đã tồn tại, chỉ cần xóa các lớp cũ và di chuyển
        map.eachLayer(layer => {
            if (!!layer.toGeoJSON) { // Xóa mọi thứ trừ tile layer
                map.removeLayer(layer);
            }
        });
        map.setView(data.start_point, 7);
    }

    // 1. Vẽ 10 điểm lịch sử (Xám)
    L.polyline(data.history_coords, { color: 'gray', weight: 3, opacity: 0.7 })
        .bindTooltip("Lịch sử (10 điểm)")
        .addTo(map);

    // 2. Điểm bắt đầu (Xanh lá)
    L.marker(data.start_point)
        .bindTooltip("Điểm 10 (Bắt đầu dự đoán)")
        .addTo(map);

    // 3. Điểm "Sự thật" (Xanh Navy)
    L.circleMarker(data.true_point, { radius: 7, color: 'navy', fillColor: 'navy', fillOpacity: 0.8 })
        .bindTooltip(`<b>Sự thật (Điểm 11)</b><br>${data.true_point.map(c => c.toFixed(2)).join(', ')}`)
        .addTo(map);
    L.polyline([data.start_point, data.true_point], { color: 'navy', weight: 2 })
        .addTo(map);

    // 4. Dự đoán PyTorch (Đỏ)
    L.circleMarker(data.pred_torch, { radius: 7, color: 'red', fillColor: 'red', fillOpacity: 0.8 })
        .bindTooltip(`<b>Dự đoán PyTorch</b><br>${data.pred_torch.map(c => c.toFixed(2)).join(', ')}`)
        .addTo(map);
    L.polyline([data.start_point, data.pred_torch], { color: 'red', weight: 2, dashArray: '5, 5' })
        .addTo(map);

    // 5. Dự đoán Scratch (Cam)
    L.circleMarker(data.pred_scratch, { radius: 7, color: 'orange', fillColor: 'orange', fillOpacity: 0.8 })
        .bindTooltip(`<b>Dự đoán Scratch</b><br>${data.pred_scratch.map(c => c.toFixed(2)).join(', ')}`)
        .addTo(map);
    L.polyline([data.start_point, data.pred_scratch], { color: 'orange', weight: 2, dashArray: '5, 5' })
        .addTo(map);
}

// Hàm để chạy khi trang web tải xong
document.addEventListener("DOMContentLoaded", function() {
    const selectBox = document.getElementById("sample-select");
    const predictButton = document.getElementById("predict-button");

    // 1. Tải danh sách Case Studies từ API
    fetch("/api/get_test_samples")
        .then(response => response.json())
        .then(samples => {
            selectBox.innerHTML = ""; // Xóa "Đang tải..."
            samples.forEach(sample => {
                const option = document.createElement("option");
                option.value = sample.id;
                option.textContent = sample.name;
                selectBox.appendChild(option);
            });
            predictButton.disabled = false; // Kích hoạt nút bấm
        })
        .catch(err => {
            console.error("Lỗi tải samples:", err);
            selectBox.innerHTML = "<option>Lỗi tải danh sách</option>";
        });

    // 2. Thêm sự kiện cho nút "Chạy Demo"
    predictButton.addEventListener("click", function() {
        const sampleId = selectBox.value;
        if (!sampleId) return;
        
        predictButton.textContent = "Đang dự đoán...";
        predictButton.disabled = true;

        // Gọi API dự đoán
        fetch("/api/predict?sample_id=${sampleId}")
            .then(response => response.json())
            .then(data => {
                // Khi có dữ liệu, gọi hàm vẽ bản đồ
                drawMap(data);
                predictButton.textContent = "Chạy Demo";
                predictButton.disabled = false;
            })
            .catch(err => {
                console.error("Lỗi dự đoán:", err);
                alert("Đã xảy ra lỗi khi dự đoán.");
                predictButton.textContent = "Chạy Demo";
                predictButton.disabled = false;
            });
    });
});