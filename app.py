import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO

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

# Streamlit UI
st.title("Bearing Defect Segmentation")

uploaded_file = st.file_uploader("Upload a .tif image", type=["tif", "jpg", "png"])

if uploaded_file:
    # Read and display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Segment"):
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 2:  # If grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:  # If RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:  # If RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Perform segmentation
        segmented_img, class_counts = segment_image(img_array)

        # Display the color legend
        st.subheader("Image with defect segmentation")
        
        # Create a string with HTML to display colors and names in a row
        legend_html = ""
        for class_id, count in class_counts.items():
            class_name = model.names[class_id]
            color = CLASS_COLORS[class_id]
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"  # Convert RGB to Hex
            legend_html += f'<span style="color:white; background-color:{color_hex}; padding:5px 10px; margin-right:10px; border-radius:5px;">{class_name} (Count: {count})</span>'

        st.markdown(legend_html, unsafe_allow_html=True)

        # Display segmented image
        st.image(segmented_img, caption="Segmented Image", use_container_width=True, channels="BGR") 