// web_app/static/js/script.js

// Biến toàn cục để giữ bản đồ
let map = null;
// (MỚI) Biến toàn cục để giữ các lớp (layers) của bản đồ tổng quan
let allTracksLayerGroup = null;
// (MỚI) Biến toàn cục để giữ lớp điều khiển (bật/tắt layer)
let layerControl = null;

// --- Hàm vẽ bản đồ cho Demo Dự đoán (Giữ nguyên) ---
// (Hàm drawMap của bạn giữ nguyên, không cần sửa)
function drawMap(data) {
    if (!map) {
        map = L.map('map').setView(data.start_point, 7);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
    } else {
        // Xóa các layer cũ (nhưng KHÔNG xóa layer tổng quan nếu có)
        map.eachLayer(layer => {
            // Chỉ xóa các layer là vector (dữ liệu), không xóa tile layer
            if (layer.options.pane === 'markerPane' || layer.options.pane === 'overlayPane') {
                 // Tránh xóa nhầm layer tổng quan
                 if (layer !== allTracksLayerGroup) {
                    map.removeLayer(layer);
                 }
            }
        });
        map.setView(data.start_point, 7);
    }

    // (MỚI) Xóa lớp tổng quan nếu nó đang hiển thị
    if (allTracksLayerGroup && map.hasLayer(allTracksLayerGroup)) {
        if (layerControl) {
            layerControl.removeLayer(allTracksLayerGroup);
        }
        map.removeLayer(allTracksLayerGroup);
        allTracksLayerGroup = null; // Đặt lại
    }

    // Vẽ các đường của demo dự đoán
    L.polyline(data.history_coords, { color: 'gray', weight: 3, opacity: 0.7 })
        .bindTooltip("History (10 points)")
        .addTo(map);
    L.marker(data.start_point)
        .bindTooltip("Point 10 (Prediction Start)")
        .addTo(map);
    L.circleMarker(data.true_point, { radius: 7, color: 'navy', fillColor: 'navy', fillOpacity: 0.8 })
        .bindTooltip(`<b>Ground Truth (Point 11)</b><br>${data.true_point.map(c => c.toFixed(2)).join(', ')}`)
        .addTo(map);
    L.polyline([data.start_point, data.true_point], { color: 'navy', weight: 2 })
        .addTo(map);
    L.circleMarker(data.pred_torch, { radius: 7, color: 'red', fillColor: 'red', fillOpacity: 0.8 })
        .bindTooltip(`<b>PyTorch Prediction</b><br>${data.pred_torch.map(c => c.toFixed(2)).join(', ')}`)
        .addTo(map);
    L.polyline([data.start_point, data.pred_torch], { color: 'red', weight: 2, dashArray: '5, 5' })
        .addTo(map);
    L.circleMarker(data.pred_scratch, { radius: 7, color: 'orange', fillColor: 'orange', fillOpacity: 0.8 })
        .bindTooltip(`<b>Scratch Prediction</b><br>${data.pred_scratch.map(c => c.toFixed(2)).join(', ')}`)
        .addTo(map);
    L.polyline([data.start_point, data.pred_scratch], { color: 'orange', weight: 2, dashArray: '5, 5' })
        .addTo(map);
}

// --- (MỚI) Các hàm Helper để vẽ bản đồ tổng quan ---

/**
 * (MỚI) Xây dựng một bảng HTML chi tiết cho *toàn bộ* cơn bão
 */
function buildStormTableHTML(storm) {
    let html = `<h4 style="margin:5px;">Storm SID: ${storm.sid}</h4>`;
    html += `<p style="margin:5px;">Total Points: ${storm.count}</p>`;
    html += '<div style="max-height: 200px; overflow-y: scroll; border: 1px solid #ccc; margin-top: 10px;">';
    html += '<table style="width:100%; font-size: 11px; border-collapse: collapse;">';
    html += '<thead style="position: sticky; top: 0; background-color: #f2f2f2;"><tr>';
    html += '<th style="padding: 4px; border: 1px solid #ddd;">Time</th>';
    html += '<th style="padding: 4px; border: 1px solid #ddd;">Lat</th>';
    html += '<th style="padding: 4px; border: 1px solid #ddd;">Lon</th>';
    html += '<th style="padding: 4px; border: 1px solid #ddd;">Wind (kt)</th>';
    html += '<th style="padding: 4px; border: 1px solid #ddd;">Pres (mb)</th>';
    html += '</tr></thead><tbody>';

    for (const p of storm.points) {
        const wind_str = p.wind ? `${p.wind.toFixed(0)}` : 'N/A';
        const pres_str = p.pres ? `${p.pres.toFixed(0)}` : 'N/A';
        html += '<tr>';
        html += `<td style="padding: 4px; border: 1px solid #ddd;">${p.time}</td>`;
        html += `<td style="padding: 4px; border: 1px solid #ddd;">${p.lat.toFixed(2)}</td>`;
        html += `<td style="padding: 4px; border: 1px solid #ddd;">${p.lon.toFixed(2)}</td>`;
        html += `<td style="padding: 4px; border: 1px solid #ddd;">${wind_str}</td>`;
        html += `<td style="padding: 4px; border: 1px solid #ddd;">${pres_str}</td>`;
        html += '</tr>';
    }
    html += '</tbody></table></div>';
    return html;
}

/**
 * (MỚI) Xây dựng một popup HTML nhỏ cho *từng điểm*
 */
function buildPointPopupHTML(p, sid) {
    const wind_str = p.wind ? `${p.wind.toFixed(0)} kt` : 'N/A';
    const pres_str = p.pres ? `${p.pres.toFixed(0)} mb` : 'N/A';
    return `
        <b>SID:</b> ${sid}<br>
        <b>Thời gian:</b> ${p.time}<br>
        <hr style="margin: 3px 0;">
        <b>Lat:</b> ${p.lat.toFixed(2)}, <b>Lon:</b> ${p.lon.toFixed(2)}<br>
        <b>Gió:</b> ${wind_str}, <b>Áp suất:</b> ${pres_str}
    `;
}

/**
 * (MỚI) Hàm chính để vẽ tất cả 399 cơn bão
 */
function drawAllTracks(storms) {
    // 1. Xóa lớp (layer) tổng quan cũ nếu nó tồn tại
    if (allTracksLayerGroup) {
        if (layerControl && map.hasLayer(allTracksLayerGroup)) {
            layerControl.removeLayer(allTracksLayerGroup);
        }
        if (map.hasLayer(allTracksLayerGroup)) {
            map.removeLayer(allTracksLayerGroup);
        }
    }

    // 2. Tạo các nhóm lớp (layer groups)
    // (BẮT BUỘC) Phải có thư viện MarkerCluster đã import trong HTML
    const markerClusterGroup = L.markerClusterGroup(); // Cho các điểm (dots)
    const polylineGroup = L.layerGroup(); // Cho các đường (lines)

    // 3. Lặp qua từng cơn bão
    for (const storm of storms) {
        const color = storm.color;
        const coords = storm.points.map(p => [p.lat, p.lon]);

        // 4. Tạo bảng HTML chi tiết (cho đường line)
        const tableHTML = buildStormTableHTML(storm);

        // 5. Tạo đường PolyLine
        const line = L.polyline(coords, {
            color: color,
            weight: 2.5,
            opacity: 0.8
        });
        line.bindPopup(tableHTML, { maxWidth: 400 });
        line.addTo(polylineGroup);

        // 6. Lặp qua từng điểm để thêm vào cluster
        for (const p of storm.points) {
            const pointPopupHTML = buildPointPopupHTML(p, storm.sid);

            const marker = L.circleMarker([p.lat, p.lon], {
                radius: 1,
                color: color,
                fill: true,
                fillColor: color,
                fillOpacity: 0.7
            });
            marker.bindPopup(pointPopupHTML);
            marker.addTo(markerClusterGroup); // <-- Thêm vào CLUSTER
        }
    }

    // 7. Nhóm cả hai lớp (lines và clusters) lại
    allTracksLayerGroup = L.layerGroup([polylineGroup, markerClusterGroup]);
    allTracksLayerGroup.addTo(map);

    // 8. Thêm Layer Control (bộ điều khiển lớp)
    if (!layerControl) {
        // Tạo bộ điều khiển nếu nó chưa tồn tại
        layerControl = L.control.layers(null, null, { collapsed: false }).addTo(map);
    }
    // Thêm lớp mới vào bộ điều khiển
    layerControl.addOverlay(allTracksLayerGroup, "Bản đồ Tổng quan (399 bão)");
}


// --- Hàm chạy khi trang được tải (SỬA ĐỔI) ---
document.addEventListener("DOMContentLoaded", function() {
    const selectBox = document.getElementById("sample-select");
    const predictButton = document.getElementById("predict-button");
    // (MỚI) Lấy nút mới và span trạng thái
    const loadAllTracksButton = document.getElementById("load-all-tracks-button");
    const loadStatus = document.getElementById("load-status");

    // 1. (MỚI) Khởi tạo bản đồ ngay lập tức
    if (!map) {
        map = L.map('map').setView([20, -40], 3); // View toàn cầu
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        // (MỚI) Tạo bộ điều khiển lớp ngay
        layerControl = L.control.layers(null, null, { collapsed: false }).addTo(map);
    }

    // 2. Tải danh sách Case Studies (Giữ nguyên)
    fetch("/api/get_test_samples")
        .then(response => response.json())
        .then(samples => {
            selectBox.innerHTML = "";
            samples.forEach(sample => {
                const option = document.createElement("option");
                option.value = sample.id;
                option.textContent = sample.name;
                selectBox.appendChild(option);
            });
            predictButton.disabled = false;
        })
        .catch(err => {
            console.error("Lỗi tải samples:", err);
            selectBox.innerHTML = "<option>Lỗi tải danh sách</option>";
        });

    // 3. Sự kiện cho nút "Run Demo" (SỬA ĐỔI)
    predictButton.addEventListener("click", function() {
        const sampleId = selectBox.value;
        if (!sampleId) return;

        predictButton.textContent = "Đang dự đoán...";
        predictButton.disabled = true;

        fetch(`/api/predict?sample_id=${sampleId}`)
            .then(response => response.json())
            .then(data => {
                // (SỬA ĐỔI) Xóa lớp tổng quan (nếu có) để xem rõ demo
                if (allTracksLayerGroup && map.hasLayer(allTracksLayerGroup)) {
                    layerControl.removeLayer(allTracksLayerGroup);
                    map.removeLayer(allTracksLayerGroup);
                }
                drawMap(data); // Vẽ bản đồ dự đoán
                predictButton.textContent = "Chạy Demo Dự đoán";
                predictButton.disabled = false;
            })
            .catch(err => {
                console.error("Lỗi dự đoán:", err);
                alert("Lỗi xảy ra khi dự đoán.");
                predictButton.textContent = "Chạy Demo Dự đoán";
                predictButton.disabled = false;
            });
    });

    // 4. (MỚI) Sự kiện cho nút "Tải Bản đồ Tổng quan"
    loadAllTracksButton.addEventListener("click", function() {
        loadStatus.textContent = "Đang tải 399 cơn bão (có thể mất vài giây)...";
        loadAllTracksButton.disabled = true;

        fetch("/api/get_all_tracks")
            .then(response => {
                if (!response.ok) {
                    // Nếu server trả về lỗi (như 500)
                    throw new Error(`Lỗi Server: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }

                // Xóa các lớp demo cũ (nếu có)
                map.eachLayer(layer => {
                    if (layer.options.pane === 'markerPane' || layer.options.pane === 'overlayPane') {
                         if (layer !== allTracksLayerGroup) { // Không xóa chính nó
                            map.removeLayer(layer);
                         }
                    }
                });

                drawAllTracks(data); // Vẽ bản đồ tổng quan
                map.setView([20, -40], 3); // Reset view
                loadStatus.textContent = `Đã tải ${data.length} cơn bão!`;
                loadAllTracksButton.disabled = false;
            })
            .catch(err => {
                console.error("Lỗi tải bản đồ tổng quan:", err);
                loadStatus.textContent = "Lỗi! Không thể tải dữ liệu.";
                alert("Lỗi tải bản đồ tổng quan: " + err.message);
                loadAllTracksButton.disabled = false;
            });
    });
});