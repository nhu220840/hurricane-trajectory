// static/js/script.js

// Global variable to hold the map
let map = null;

// Function to draw the map (will be called when data is available)
function drawMap(data) {
    // If the map hasn't been created, create it
    if (!map) {
        map = L.map('map').setView(data.start_point, 7); // Zoom in on the start point
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
    } else {
        // If the map already exists, just clear old layers and move
        map.eachLayer(layer => {
            if (!!layer.toGeoJSON) { // Remove everything except the tile layer
                map.removeLayer(layer);
            }
        });
        map.setView(data.start_point, 7);
    }

    // 1. Draw 10 history points (Gray)
    L.polyline(data.history_coords, { color: 'gray', weight: 3, opacity: 0.7 })
        .bindTooltip("History (10 points)")
        .addTo(map);

    // 2. Start point (Green)
    L.marker(data.start_point)
        .bindTooltip("Point 10 (Prediction Start)")
        .addTo(map);

    // 3. "Ground Truth" point (Navy)
    L.circleMarker(data.true_point, { radius: 7, color: 'navy', fillColor: 'navy', fillOpacity: 0.8 })
        .bindTooltip(`<b>Ground Truth (Point 11)</b><br>${data.true_point.map(c => c.toFixed(2)).join(', ')}`)
        .addTo(map);
    L.polyline([data.start_point, data.true_point], { color: 'navy', weight: 2 })
        .addTo(map);

    // 4. PyTorch Prediction (Red)
    L.circleMarker(data.pred_torch, { radius: 7, color: 'red', fillColor: 'red', fillOpacity: 0.8 })
        .bindTooltip(`<b>PyTorch Prediction</b><br>${data.pred_torch.map(c => c.toFixed(2)).join(', ')}`)
        .addTo(map);
    L.polyline([data.start_point, data.pred_torch], { color: 'red', weight: 2, dashArray: '5, 5' })
        .addTo(map);

    // 5. Scratch Prediction (Orange)
    L.circleMarker(data.pred_scratch, { radius: 7, color: 'orange', fillColor: 'orange', fillOpacity: 0.8 })
        .bindTooltip(`<b>Scratch Prediction</b><br>${data.pred_scratch.map(c => c.toFixed(2)).join(', ')}`)
        .addTo(map);
    L.polyline([data.start_point, data.pred_scratch], { color: 'orange', weight: 2, dashArray: '5, 5' })
        .addTo(map);
}

// Function to run when the web page is loaded
document.addEventListener("DOMContentLoaded", function() {
    const selectBox = document.getElementById("sample-select");
    const predictButton = document.getElementById("predict-button");

    // 1. Load the list of Case Studies from the API
    fetch("/api/get_test_samples")
        .then(response => response.json())
        .then(samples => {
            selectBox.innerHTML = ""; // Clear "Loading..."
            samples.forEach(sample => {
                const option = document.createElement("option");
                option.value = sample.id;
                option.textContent = sample.name;
                selectBox.appendChild(option);
            });
            predictButton.disabled = false; // Enable the button
        })
        .catch(err => {
            console.error("Error loading samples:", err);
            selectBox.innerHTML = "<option>Error loading list</option>";
        });

    // 2. Add event listener for the "Run Demo" button
    predictButton.addEventListener("click", function() {
        const sampleId = selectBox.value;
        if (!sampleId) return;

        predictButton.textContent = "Predicting...";
        predictButton.disabled = true;

        // Call prediction API
        fetch(`/api/predict?sample_id=${sampleId}`) // Corrected template literal
            .then(response => response.json())
            .then(data => {
                // When data is received, call the map drawing function
                drawMap(data);
                predictButton.textContent = "Run Demo";
                predictButton.disabled = false;
            })
            .catch(err => {
                console.error("Prediction error:", err);
                alert("An error occurred during prediction.");
                predictButton.textContent = "Run Demo";
                predictButton.disabled = false;
            });
    });
});