from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
from ultralytics import YOLO
from openai import OpenAI
from dotenv import load_dotenv
import io
import os
import torch
import base64
import cv2
import numpy as np
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom.filebase import DicomBytesIO


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI setup
app = FastAPI(title="API for anomaly detection in mammographic images, by Edgar Garcia, Gabriela Bula, Lena Castillo")

# CORS
origins = ["http://localhost", "http://localhost:8080", "http://127.0.0.1", "http://127.0.0.1:8080", "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
try:
    model = YOLO("app/best.pt")
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading best.pt: {e}")


class DicomProcessor:
    def __init__(self, voi_lut=True, clahe_clip=2.0, clahe_grid=(8, 8)):
        self.voi_lut = voi_lut
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    
    def process(self, dicom_data, photometric_interpretation=None):
        """
        Returns processed image from dicom data.
        Fixes monochrome and applies CLAHE filter for segmentation.
        
        Based on: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
        """
        if isinstance(dicom_data, bytes):
            dicom = dcmread(DicomBytesIO(dicom_data))
        else:
            dicom = dicom_data
        
        # Get photometric interpretation from DICOM if not provided
        if photometric_interpretation is None:
            photometric_interpretation = getattr(dicom, 'PhotometricInterpretation', 'MONOCHROME2')
        
        # Transforms raw DICOM data to "human-friendly" view
        data = apply_voi_lut(dicom.pixel_array, dicom) if self.voi_lut else dicom.pixel_array
        
        # Inverts X-ray if needed
        if photometric_interpretation == "MONOCHROME1":
            data = np.amax(data) - data
        
        # Normalization
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
        
        # Apply CLAHE filter
        clahe = self.clahe.apply(data.astype(np.uint8))
        
        return clahe


def dicom_to_pil_image(dicom_bytes, target_size=(640, 640)):
    """
    Convert DICOM bytes to PIL Image
    """
    processor = DicomProcessor()
    
    # Process DICOM
    processed_array = processor.process(dicom_bytes)
    
    # Resize if needed
    if target_size:
        processed_array = cv2.resize(processed_array, target_size)
    
    # Convert to PIL Image
    # OpenCV uses BGR, PIL uses RGB, but for grayscale it doesn't matter
    pil_image = Image.fromarray(processed_array, mode='L')  # 'L' for grayscale
    
    # Convert to RGB if needed for the model
    pil_image = pil_image.convert('RGB')
    
    return pil_image


@app.get("/")
def raiz():
    return {"message": "API By Edgar Garcia, Gabriela Bula, Lena Castillo"}


@app.post("/predict/")
async def predict(file: UploadFile = File(None), question: str = Form(...)):
    annotated_image_b64 = None
    detection_result = None
    class_label = None

    # If an image is uploaded, analyze it
    if file:
        if not model:
            return {"error": "Model not loaded. Please check the server logs."}

        content = await file.read()
        
        # Check if file is DICOM or regular image
        if file.filename.lower().endswith('.dicom') or file.filename.lower().endswith('.dcm'):
            # Process DICOM file
            try:
                image = dicom_to_pil_image(content)
            except Exception as e:
                return {"error": f"DICOM file could not be processed: {e}"}
        else:
            # Handle regular image files
            if not file.content_type.startswith("image/"):
                return {"error": "File type is not an image or DICOM file."}
            
            try:
                image = Image.open(io.BytesIO(content))
            except Exception as e:
                return {"error": f"Image could not be loaded: {e}"}

        # Run YOLO inference
        try:
            results = model(image)
        except Exception as e:
            return {"error": f"Error in model inference: {e}"}

        detections = []
        # Create a copy for annotation
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        res = results[0]
        class_names = res.names

        for box in res.boxes:
            cls = int(box.cls)
            conf = round(float(box.conf), 4)
            bbox = [round(c.item(), 2) for c in box.xyxy[0]]
            name = class_names[cls]

            detections.append({
                "class": cls,
                "confidence": conf,
                "bbox_xyxy": bbox,
                "className": name
            })

            # Draw bounding box
            draw.rectangle(bbox, outline="red", width=2)
            label = f"{name} ({conf:.2f})"
            y_offset = bbox[1] - 15 if bbox[1] - 15 > 5 else bbox[1] + 5
            draw.text((bbox[0], y_offset), label, fill="red")

        # Encode annotated image
        buffer = io.BytesIO()
        annotated_image.save(buffer, format="JPEG")
        annotated_image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Interpret detection results
        if detections:
            class_label = detections[0]['class']
            if class_label == 0:
                detection_result = "Anomalies were detected in the mammographic image."
            elif class_label == 1:
                detection_result = "No anomalies were detected in the mammographic image."
            else:
                detection_result = "Detection result is unclear."
        else:
            detection_result = "No objects were detected in the image."

        # Generate contextual explanation with OpenAI
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in explaining mammography and medical imaging results."},
                    {"role": "user", "content": f"{detection_result} Based on this result, please answer: {question}"}
                ],
                max_tokens=500,
                temperature=0.7
            )
            explanation = response.choices[0].message.content
        except Exception as e:
            explanation = f"Error generating explanation: {e}"

        return {
            "results": detections,
            "annotated_image_base64": annotated_image_b64,
            "detection_result": detection_result,
            "explanation": explanation,
            "file_type": "dicom" if file.filename.lower().endswith(('.dicom', '.dcm')) else "image"
        }

    # If no image uploaded, answer general question
    else:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical assistant knowledgeable in mammographic anomaly detection."},
                    {"role": "user", "content": question}
                ],
                max_tokens=500,
                temperature=0.7
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error generating response from OpenAI: {e}"

        return {"response": answer}



