
import gradio as gr
from ultralytics import YOLO
from PIL import Image
import os

# Step 1: Load the trained YOLO model
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = YOLO(MODEL_PATH)


# Step 2: Create prediction function
def detect_objects(image, confidence):
    if image is None:
        return None, "Please upload an image."

    # Run YOLO prediction
    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )

    # Draw bounding boxes on the image
    result_image = results[0].plot()

    # Convert BGR image to RGB for Gradio
    result_image = result_image[:, :, ::-1]

    # Create a readable detection summary
    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        score = float(box.conf[0])

        detections.append(
            f"{class_name}: {score:.2%}"
        )

    if detections:
        summary = "\n".join(detections)
    else:
        summary = "No objects detected."

    return Image.fromarray(result_image), summary


# Step 3: Create the web interface
demo = gr.Interface(
    fn=detect_objects,
    inputs=[
        gr.Image(
            type="pil",
            label="Upload an image"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=0.9,
            value=0.25,
            step=0.05,
            label="Confidence Threshold"
        )
    ],
    outputs=[
        gr.Image(label="Detection Result"),
        gr.Textbox(label="Detected Objects")
    ],
    title="Road Traffic Detection using YOLO",
    description=(
        "Upload an image to detect road traffic objects."
    )
)


# Step 4: Launch the app
if __name__ == "__main__":
    demo.launch()