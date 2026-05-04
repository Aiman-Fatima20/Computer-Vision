const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture');
const webcamButton = document.getElementById('webcam-button');
const uploadForm = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const webcamContainer = document.getElementById('webcam-container');
const beautyPercentageDisplay = document.getElementById('beauty-percentage');
const capturedFrame = document.getElementById('captured-frame');
const uploadedImage = document.getElementById('uploaded-image');
const uploadedImageContainer = document.getElementById('uploaded-image-container');

let stream = null;

// Function to start the webcam
function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(s => {
            stream = s;
            video.srcObject = stream;
            webcamContainer.style.display = 'block';
            webcamButton.textContent = 'Stop Webcam';
            // Hide the uploaded image when starting the webcam
            uploadedImageContainer.style.display = 'none';
        })
        .catch(err => {
            console.error("Error accessing the webcam: " + err);
            alert("Could not access the webcam. Please check permissions.");
        });
}

// Function to stop the webcam
function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        video.srcObject = null;
        webcamContainer.style.display = 'none';
        webcamButton.textContent = 'Start Webcam';
        // Hide the captured frame when stopping the webcam
        capturedFrame.style.display = 'none';
    }
}

// Toggle webcam on/off
webcamButton.addEventListener('click', () => {
    if (webcamButton.textContent === 'Start Webcam') {
        startWebcam();
    } else {
        stopWebcam();
    }
});

// Handle webcam capture
captureButton.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageDataURL = canvas.toDataURL('image/png');

    // Display the captured frame
    capturedFrame.src = imageDataURL;
    capturedFrame.style.display = 'block';

    // Send the captured image to the server
    fetch('/upload', {
        method: 'POST',
        body: JSON.stringify({ image: imageDataURL }),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        beautyPercentageDisplay.textContent = data.beauty_percentage;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error processing the image.');
    });
});

// Handle file upload
fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (file) {
        // Display the uploaded image
        const reader = new FileReader();
        reader.onload = function(e) {
            uploadedImage.src = e.target.result;
            uploadedImageContainer.style.display = 'block';
        };
        reader.readAsDataURL(file);
        // Hide the captured frame when uploading an image
        capturedFrame.style.display = 'none';
    }

    const formData = new FormData(uploadForm);
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        beautyPercentageDisplay.textContent = data.beauty_percentage;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error uploading the image.');
    });
});