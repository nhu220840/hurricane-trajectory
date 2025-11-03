// Global variable to hold the map
let map = null;
// (NEW) Global variable to hold the overview map layers
let allTracksLayerGroup = null;
// (NEW) Global variable to hold the layer control (toggle layers)
let layerControl = null;

// --- Map drawer for the Prediction Demo (unchanged structure) ---
function drawMap(data) {
    if (!map) {
        map = L.map('map').setView(data.start_point, 7);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
    } else {
        // Remove old vector layers (but DO NOT remove the overview layer if present)
        map.eachLayer(layer => {
            if (layer.options.pane === 'markerPane' || layer.options.pane === 'overlayPane') {
                 if (layer !== allTracksLayerGroup) {
                    map.removeLayer(layer);
                 }
            }
        });
        map.setView(data.start_point, 7);
    }

    // (NEW) Remove overview layer if it is currently displayed
    if (allTracksLayerGroup && map.hasLayer(allTracksLayerGroup)) {
        if (layerControl) {
            layerControl.removeLayer(allTracksLayerGroup);
        }
        map.removeLayer(allTracksLayerGroup);
        allTracksLayerGroup = null;
    }

    // Draw prediction demo tracks
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

// --- (NEW) Helper functions for the overview map ---

/**
 * (NEW) Build a detailed HTML table for the entire storm
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
    html += '<th style="padding: 4px; border: 1px solid #ddd;">Pressure (mb)</th>';
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
 * (NEW) Build a small popup HTML for each point
 */
function buildPointPopupHTML(p, sid) {
    const wind_str = p.wind ? `${p.wind.toFixed(0)} kt` : 'N/A';
    const pres_str = p.pres ? `${p.pres.toFixed(0)} mb` : 'N/A';
    return `
        <b>SID:</b> ${sid}<br>
        <b>Time:</b> ${p.time}<br>
        <hr style="margin: 3px 0;">
        <b>Lat:</b> ${p.lat.toFixed(2)}, <b>Lon:</b> ${p.lon.toFixed(2)}<br>
        <b>Wind:</b> ${wind_str}, <b>Pressure:</b> ${pres_str}
    `;
}

/**
 * (NEW) Main function to draw all 399 storms
 */
function drawAllTracks(storms) {
    // 1. Remove previous overview layer if it exists
    if (allTracksLayerGroup) {
        if (layerControl && map.hasLayer(allTracksLayerGroup)) {
            layerControl.removeLayer(allTracksLayerGroup);
        }
        if (map.hasLayer(allTracksLayerGroup)) {
            map.removeLayer(allTracksLayerGroup);
        }
    }

    // 2. Create layer groups (requires MarkerCluster in HTML)
    const markerClusterGroup = L.markerClusterGroup(); // For points (dots)
    const polylineGroup = L.layerGroup(); // For polylines (tracks)

    // 3. Iterate through storms
    for (const storm of storms) {
        const color = storm.color;
        const coords = storm.points.map(p => [p.lat, p.lon]);

        // 4. Build detailed HTML table (for the polyline popup)
        const tableHTML = buildStormTableHTML(storm);

        // 5. Create PolyLine
        const line = L.polyline(coords, {
            color: color,
            weight: 2.5,
            opacity: 0.8
        });
        line.bindPopup(tableHTML, { maxWidth: 400 });
        line.addTo(polylineGroup);

        // 6. Add each point to the cluster
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
            marker.addTo(markerClusterGroup);
        }
    }

    // 7. Group both layers (lines and clusters)
    allTracksLayerGroup = L.layerGroup([polylineGroup, markerClusterGroup]);
    allTracksLayerGroup.addTo(map);

    // 8. Add Layer Control
    if (!layerControl) {
        layerControl = L.control.layers(null, null, { collapsed: false }).addTo(map);
    }
    layerControl.addOverlay(allTracksLayerGroup, "Overview Map (399 storms)");
}


// --- Run when the page loads ---
document.addEventListener("DOMContentLoaded", function() {
    const selectBox = document.getElementById("sample-select");
    const predictButton = document.getElementById("predict-button");
    // (NEW) Get the new button and status span
    const loadAllTracksButton = document.getElementById("load-all-tracks-button");
    const loadStatus = document.getElementById("load-status");

    // 1. (NEW) Initialize the map immediately
    if (!map) {
        map = L.map('map').setView([20, -40], 3); // Global view
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        // (NEW) Create the layer control immediately
        layerControl = L.control.layers(null, null, { collapsed: false }).addTo(map);
    }

    // 2. Load list of Case Studies
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
            console.error("Failed to load samples:", err);
            selectBox.innerHTML = "<option>Failed to load list</option>";
        });

    // 3. Event for "Run Demo" button
    predictButton.addEventListener("click", function() {
        const sampleId = selectBox.value;
        if (!sampleId) return;

        predictButton.textContent = "Predicting...";
        predictButton.disabled = true;

        fetch(`/api/predict?sample_id=${sampleId}`)
            .then(response => response.json())
            .then(data => {
                // Remove overview layer (if any) to make the demo clear
                if (allTracksLayerGroup && map.hasLayer(allTracksLayerGroup)) {
                    layerControl.removeLayer(allTracksLayerGroup);
                    map.removeLayer(allTracksLayerGroup);
                }
                drawMap(data); // Draw prediction map
                predictButton.textContent = "Run Prediction Demo";
                predictButton.disabled = false;
            })
            .catch(err => {
                console.error("Prediction error:", err);
                alert("An error occurred during prediction.");
                predictButton.textContent = "Run Prediction Demo";
                predictButton.disabled = false;
            });
    });

    // 4. Event for "Load Overview Map" button
    loadAllTracksButton.addEventListener("click", function() {
        loadStatus.textContent = "Loading 399 storms (this may take a few seconds)...";
        loadAllTracksButton.disabled = true;

        fetch("/api/get_all_tracks")
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server error: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }

                // Remove old demo layers (if any)
                map.eachLayer(layer => {
                    if (layer.options.pane === 'markerPane' || layer.options.pane === 'overlayPane') {
                         if (layer !== allTracksLayerGroup) {
                            map.removeLayer(layer);
                         }
                    }
                });

                drawAllTracks(data); // Draw overview map
                map.setView([20, -40], 3); // Reset view
                loadStatus.textContent = `Loaded ${data.length} storms!`;
                loadAllTracksButton.disabled = false;
            })
            .catch(err => {
                console.error("Failed to load overview map:", err);
                loadStatus.textContent = "Error! Could not load data.";
                alert("Failed to load overview map: " + err.message);
                loadAllTracksButton.disabled = false;
            });
    });
});
