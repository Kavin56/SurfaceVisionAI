from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize YOLO model
model = YOLO("model/best.pt")

# Fixed color mapping for each class (BGR)
CLASS_COLORS = {
    0: (0, 0, 255),    # Shallow Groove - Blue
    1: (0, 255, 0),    # Pit - Green
    2: (255, 0, 0),    # Deep Groove - Red
}

def preprocess_image(img):
    img = cv2.resize(img, (640, 640))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return img

def segment_image(img):
    processed_img = preprocess_image(img)
    results = model.predict(processed_img, imgsz=640, save=False)[0]

    overlay = processed_img.copy()
    masks = results.masks
    classes = results.boxes.cls.cpu().numpy().astype(int)

    # Count occurrences of each class
    class_counts = {class_id: 0 for class_id in CLASS_COLORS}

    if masks is not None:
        masks_data = masks.data.cpu().numpy()

        for i, mask in enumerate(masks_data):
            class_id = classes[i]
            class_counts[class_id] += 1

            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            binary_mask = (mask > 0.5).astype(np.uint8)

            # Create a colored mask
            colored_mask = np.zeros_like(processed_img, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = binary_mask * color[c]

            # Blend with transparency
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.25, 0)

    return overlay, class_counts

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the image from the request
        file = request.files['image']
        
        # Read the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert PIL Image to OpenCV format
        img_array = np.array(img)
        if len(img_array.shape) == 2:  # If grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:  # If RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:  # If RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Process the image
        segmented_img, class_counts = segment_image(img_array)

        # Convert both original and processed images to base64
        _, buffer_original = cv2.imencode('.png', img_array)
        original_img_base64 = base64.b64encode(buffer_original).decode('utf-8')

        _, buffer_processed = cv2.imencode('.png', segmented_img)
        processed_img_base64 = base64.b64encode(buffer_processed).decode('utf-8')

        # Prepare the response
        response = {
            'original_image': original_img_base64,
            'processed_image': processed_img_base64,
            'defect_counts': {
                'SHALLOW_GROOVE': int(class_counts.get(0, 0)),
                'PIT': int(class_counts.get(1, 0)),
                'DEEP_GROOVE': int(class_counts.get(2, 0))
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 