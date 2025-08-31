// Handle file upload display
document.getElementById("audio-upload").addEventListener("change", function () {
    const file = this.files[0];
    if (file) {
        document.getElementById("emotion-output").innerText = "File uploaded: " + file.name;
    }
});

// Handle audio evaluation
function evaluateAudio() {
    const file = document.getElementById("audio-upload").files[0];

    if (!file) {
        alert("Please upload an audio file first.");
        return;
    }

    const formData = new FormData();
    formData.append("audio", file);

    fetch("http://localhost:5000/predict", { // Replace with your actual Flask endpoint
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById("emotion-output").innerText = "Error: " + data.error;
            } else {
                document.getElementById("emotion-output").innerText = "Detected Emotion: " + data.emotion + " (" + (data.confidence * 100).toFixed(2) + "%)";
            }
        })
        .catch(error => {
            console.error("Error:", error);
            document.getElementById("emotion-output").innerText = "Error processing file.";
        });
}