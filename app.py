
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Function to calculate beauty percentage using facial landmarks
def calculate_beauty_percentage(image):
    # Convert image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Face Mesh
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return 0  # No face detected
    
    # Get the landmarks (MediaPipe provides 468 points)
    landmarks = results.multi_face_landmarks[0].landmark
    (h, w) = image.shape[:2]
    
    # Convert normalized landmarks to pixel coordinates
    landmarks_pixels = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        landmarks_pixels.append((x, y))
    
    # Select 50 landmarks for beauty calculation
    # Eyes (8 landmarks)
    left_eye_center = landmarks_pixels[33]      # Left eye center
    left_eye_inner = landmarks_pixels[130]      # Left eye inner corner
    left_eye_outer = landmarks_pixels[159]      # Left eye outer corner
    left_eye_bottom = landmarks_pixels[145]     # Left eye bottom
    right_eye_center = landmarks_pixels[263]    # Right eye center
    right_eye_inner = landmarks_pixels[359]     # Right eye inner corner
    right_eye_outer = landmarks_pixels[386]     # Right eye outer corner
    right_eye_bottom = landmarks_pixels[374]    # Right eye bottom
    
    # Eyebrows (10 landmarks)
    left_eyebrow_inner = landmarks_pixels[52]   # Left eyebrow inner
    left_eyebrow_middle = landmarks_pixels[55]  # Left eyebrow middle
    left_eyebrow_outer = landmarks_pixels[65]   # Left eyebrow outer
    left_eyebrow_1 = landmarks_pixels[53]       # Additional left eyebrow point
    left_eyebrow_2 = landmarks_pixels[63]       # Additional left eyebrow point
    right_eyebrow_inner = landmarks_pixels[282] # Right eyebrow inner
    right_eyebrow_middle = landmarks_pixels[285] # Right eyebrow middle
    right_eyebrow_outer = landmarks_pixels[295]  # Right eyebrow outer
    right_eyebrow_1 = landmarks_pixels[283]     # Additional right eyebrow point
    right_eyebrow_2 = landmarks_pixels[293]     # Additional right eyebrow point
    
    # Nose (6 landmarks)
    nose_tip = landmarks_pixels[1]              # Nose tip
    nose_bridge_1 = landmarks_pixels[6]         # Nose bridge
    nose_bridge_2 = landmarks_pixels[4]         # Nose bridge higher
    nose_base_1 = landmarks_pixels[2]           # Nose base center
    nose_base_left = landmarks_pixels[94]       # Nose base left
    nose_base_right = landmarks_pixels[195]     # Nose base right
    
    # Mouth (10 landmarks)
    left_mouth = landmarks_pixels[61]           # Left mouth corner
    right_mouth = landmarks_pixels[291]         # Right mouth corner
    upper_lip_center = landmarks_pixels[13]     # Upper lip center
    upper_lip_left = landmarks_pixels[40]       # Upper lip left
    upper_lip_right = landmarks_pixels[270]     # Upper lip right
    lower_lip_center = landmarks_pixels[14]     # Lower lip center
    lower_lip_left = landmarks_pixels[17]       # Lower lip left
    lower_lip_right = landmarks_pixels[178]     # Lower lip right
    mouth_inner_left = landmarks_pixels[78]     # Mouth inner left
    mouth_inner_right = landmarks_pixels[308]   # Mouth inner right
    
    # Chin and Jawline (10 landmarks)
    chin = landmarks_pixels[199]                # Chin
    left_jaw_1 = landmarks_pixels[172]          # Left jaw near chin
    right_jaw_1 = landmarks_pixels[397]         # Right jaw near chin
    left_jaw_2 = landmarks_pixels[148]          # Left jaw
    left_jaw_3 = landmarks_pixels[176]          # Left jaw
    right_jaw_2 = landmarks_pixels[377]         # Right jaw
    right_jaw_3 = landmarks_pixels[400]         # Right jaw
    jaw_center_1 = landmarks_pixels[152]        # Jaw center lower
    jaw_center_2 = landmarks_pixels[378]        # Jaw center higher
    jaw_center_3 = landmarks_pixels[206]        # Jaw center near chin
    
    # Forehead (4 landmarks)
    forehead_center = landmarks_pixels[10]      # Forehead center
    forehead_1 = landmarks_pixels[9]            # Forehead slightly lower
    forehead_2 = landmarks_pixels[8]            # Forehead higher
    forehead_3 = landmarks_pixels[151]          # Forehead near hairline
    
    # Face Contours (2 landmarks)
    left_contour = landmarks_pixels[127]        # Left face contour
    right_contour = landmarks_pixels[356]       # Right face contour
    
    # Calculate beauty metrics
    # 1. Eye symmetry (distance from eye centers to nose tip)
    left_eye_to_nose = np.sqrt((left_eye_center[0] - nose_tip[0])**2 + (left_eye_center[1] - nose_tip[1])**2)
    right_eye_to_nose = np.sqrt((right_eye_center[0] - nose_tip[0])**2 + (right_eye_center[1] - nose_tip[1])**2)
    eye_symmetry = abs(left_eye_to_nose - right_eye_to_nose)
    eye_symmetry_score = 100 - (eye_symmetry / max(left_eye_to_nose, right_eye_to_nose) * 100)
    
    # 2. Eyebrow symmetry (distance from eyebrow middles to nose bridge)
    left_eyebrow_to_nose = np.sqrt((left_eyebrow_middle[0] - nose_bridge_1[0])**2 + (left_eyebrow_middle[1] - nose_bridge_1[1])**2)
    right_eyebrow_to_nose = np.sqrt((right_eyebrow_middle[0] - nose_bridge_1[0])**2 + (right_eyebrow_middle[1] - nose_bridge_1[1])**2)
    eyebrow_symmetry = abs(left_eyebrow_to_nose - right_eyebrow_to_nose)
    eyebrow_symmetry_score = 100 - (eyebrow_symmetry / max(left_eyebrow_to_nose, right_eyebrow_to_nose) * 100)
    
    # 3. Mouth symmetry (distance from mouth corners to nose base)
    left_mouth_to_nose = np.sqrt((left_mouth[0] - nose_base_1[0])**2 + (left_mouth[1] - nose_base_1[1])**2)
    right_mouth_to_nose = np.sqrt((right_mouth[0] - nose_base_1[0])**2 + (right_mouth[1] - nose_base_1[1])**2)
    mouth_symmetry = abs(left_mouth_to_nose - right_mouth_to_nose)
    mouth_symmetry_score = 100 - (mouth_symmetry / max(left_mouth_to_nose, right_mouth_to_nose) * 100)
    
    # 4. Facial width-to-height ratio (jaw width to forehead-to-chin height)
    jaw_width = np.sqrt((left_jaw_1[0] - right_jaw_1[0])**2 + (left_jaw_1[1] - right_jaw_1[1])**2)
    face_height = np.sqrt((forehead_center[0] - chin[0])**2 + (forehead_center[1] - chin[1])**2)
    ideal_wh_ratio = 1.5
    actual_wh_ratio = jaw_width / face_height
    wh_score = 100 - (abs(actual_wh_ratio - ideal_wh_ratio) / ideal_wh_ratio * 100)
    
    # 5. Nose-to-chin proportion (nose tip to chin vs. eye distance)
    eye_distance = np.sqrt((right_eye_center[0] - left_eye_center[0])**2 + (right_eye_center[1] - left_eye_center[1])**2)
    nose_to_chin = np.sqrt((nose_tip[0] - chin[0])**2 + (nose_tip[1] - chin[1])**2)
    ideal_nose_ratio = 2.0
    actual_nose_ratio = nose_to_chin / eye_distance
    nose_proportion_score = 100 - (abs(actual_nose_ratio - ideal_nose_ratio) / ideal_nose_ratio * 100)
    
    # 6. Jawline symmetry (distance from jaw points to chin)
    left_jaw_to_chin = np.sqrt((left_jaw_1[0] - chin[0])**2 + (left_jaw_1[1] - chin[1])**2)
    right_jaw_to_chin = np.sqrt((right_jaw_1[0] - chin[0])**2 + (right_jaw_1[1] - chin[1])**2)
    jaw_symmetry = abs(left_jaw_to_chin - right_jaw_to_chin)
    jaw_symmetry_score = 100 - (jaw_symmetry / max(left_jaw_to_chin, right_jaw_to_chin) * 100)
    
    # Combine scores with weighted contributions
    beauty_score = (
        eye_symmetry_score * 0.2 +      # 20% weight
        eyebrow_symmetry_score * 0.2 +  # 20% weight
        mouth_symmetry_score * 0.15 +   # 15% weight
        wh_score * 0.15 +               # 15% weight
        nose_proportion_score * 0.15 +  # 15% weight
        jaw_symmetry_score * 0.15       # 15% weight
    )
    return max(0, min(100, int(beauty_score)))

# Route to serve the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image uploads and webcam captures
@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' in request.files:
            file = request.files['file']
            image = np.array(Image.open(file))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif request.is_json:
            data = request.get_json()
            image_data = data['image']
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = np.array(Image.open(BytesIO(image_bytes)))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            return jsonify({'error': 'No image provided'}), 400

        beauty_percentage = calculate_beauty_percentage(image)
        return jsonify({'beauty_percentage': beauty_percentage})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)