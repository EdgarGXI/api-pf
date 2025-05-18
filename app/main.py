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
from ensemble_boxes import weighted_boxes_fusion
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

# Load YOLO models for ensemble
try:
    model_paths = ["app/best.pt", "app/best2.pt", "app/best3.pt"]
    models = []
    
    for model_path in model_paths:
        try:
            model = YOLO(model_path)
            models.append(model)
            print(f"Model {model_path} loaded successfully.")
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
    
    if not models:
        print("No models could be loaded.")
        models = None
    else:
        print(f"Ensemble of {len(models)} models loaded successfully.")
        
except Exception as e:
    models = None
    print(f"Error loading models: {e}")


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


def ensemble_predict_wbf_single_image(models, image, iou_threshold=0.5, confidence_threshold=0.25, skip_box_thr=0.0001):
    """
    Performs ensemble prediction using Weighted Box Fusion on multiple YOLO models for a single image.
    
    Returns fused detections and classification result
    """
    # Get predictions from each model
    all_boxes = []
    all_scores = []
    all_classes = []
    
    # Store individual model results for detailed analysis
    individual_results = []
    
    for i, model in enumerate(models):
        try:
            results = model.predict(image, conf=confidence_threshold, verbose=False)[0]
            individual_results.append(results)
            
            if len(results.boxes) > 0:
                # Get image dimensions
                img_height, img_width = results.orig_shape[:2]
                
                # Extract boxes, scores and classes
                # Convert to relative coordinates (0-1 range)
                boxes = results.boxes.xyxy.cpu().numpy()
                relative_boxes = boxes / np.array([img_width, img_height, img_width, img_height])
                
                all_boxes.append(relative_boxes.tolist())
                all_scores.append(results.boxes.conf.cpu().numpy().tolist())
                all_classes.append(results.boxes.cls.cpu().numpy().tolist())
            else:
                # Add empty lists if no detections
                all_boxes.append([])
                all_scores.append([])
                all_classes.append([])
                
        except Exception as e:
            print(f"Error in model {i}: {e}")
            # Add empty lists on error
            all_boxes.append([])
            all_scores.append([])
            all_classes.append([])
    
    # If no detections from any model
    if all(len(boxes) == 0 for boxes in all_boxes):
        return [], 1, individual_results  # 1 = no anomaly detected
    
    # Apply Weighted Box Fusion only if there are detections
    try:
        fused_boxes, fused_scores, fused_classes = weighted_boxes_fusion(
            all_boxes, all_scores, all_classes, 
            iou_thr=iou_threshold, 
            skip_box_thr=skip_box_thr
        )
        
        # Convert back to absolute coordinates if we have detections
        if len(fused_boxes) > 0 and len(individual_results) > 0:
            img_height, img_width = individual_results[0].orig_shape[:2]
            fused_boxes = fused_boxes * np.array([img_width, img_height, img_width, img_height])
        
        # Create detection results in the expected format
        detections = []
        for i, (box, score, cls) in enumerate(zip(fused_boxes, fused_scores, fused_classes)):
            detections.append({
                "class": int(cls),
                "confidence": round(float(score), 4),
                "bbox_xyxy": [round(float(coord), 2) for coord in box],
                "className": "anomaly" if int(cls) == 0 else "normal"
            })
        
        # Determine classification: 0 = anomaly detected, 1 = no anomaly
        classification_result = 0 if len(detections) > 0 else 1
        
        return detections, classification_result, individual_results
        
    except Exception as e:
        print(f"Error in weighted box fusion: {e}")
        # Fallback: return results from the first model that had detections
        for results in individual_results:
            if len(results.boxes) > 0:
                detections = []
                for box in results.boxes:
                    cls = int(box.cls)
                    conf = round(float(box.conf), 4)
                    bbox = [round(c.item(), 2) for c in box.xyxy[0]]
                    name = "anomaly" if cls == 0 else "normal"
                    
                    detections.append({
                        "class": cls,
                        "confidence": conf,
                        "bbox_xyxy": bbox,
                        "className": name
                    })
                return detections, 0, individual_results
        
        # If all fails, return no detections
        return [], 1, individual_results


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
        if not models:
            return {"error": "Models not loaded. Please check the server logs."}

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

        # Run ensemble inference
        try:
            detections, class_label, individual_results = ensemble_predict_wbf_single_image(models, image)
        except Exception as e:
            return {"error": f"Error in ensemble inference: {e}"}

        # Create a copy for annotation
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        # Draw bounding boxes from ensemble results
        for detection in detections:
            bbox = detection["bbox_xyxy"]
            conf = detection["confidence"]
            name = detection["className"]

            # Draw bounding box
            draw.rectangle(bbox, outline="red", width=3)
            label = f"{name} ({conf:.2f})"
            y_offset = bbox[1] - 15 if bbox[1] - 15 > 5 else bbox[1] + 5
            draw.text((bbox[0], y_offset), label, fill="red")


        # Encode annotated image
        buffer = io.BytesIO()
        annotated_image.save(buffer, format="JPEG")
        annotated_image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Interpret detection results
        if detections:
            if class_label == 0:
                detection_result = f"Anomalies were detected in the mammographic image"
            else:
                detection_result = f"No anomalies were detected in the mammographic image"
        else:
            detection_result = f"No objects were detected in the image"

        # Generate contextual explanation with OpenAI
        try:
            # Include ensemble information in the context

            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in explaining mammography and medical imaging results. You understand machine learning"},
                    {"role": "user", "content": f"{detection_result} Based on this result, please answer: {question}"}
                ],
                max_tokens=500,
                temperature=0.5
            )
            explanation = response.choices[0].message.content
        except Exception as e:
            explanation = f"Error generating explanation: {e}"

        return {
            "results": detections,
            "annotated_image_base64": annotated_image_b64,
            "detection_result": detection_result,
            "explanation": explanation,
            "file_type": "dicom" if file.filename.lower().endswith(('.dicom', '.dcm')) else "image",
            "ensemble_info": {
                "models_used": len(models),
                "model_paths": [path.split('/')[-1] for path in ["app/best.pt", "app/best2.pt", "app/best3.pt"] if models],
                "fusion_method": "Weighted Box Fusion"
            }
        }

    # If no image uploaded, answer general question
    else:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical assistant knowledgeable in mammographic anomaly detection and ensemble learning methods."},
                    {"role": "user", "content": question}
                ],
                max_tokens=500,
                temperature=0.7
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error generating response from OpenAI: {e}"

        return {
            "response": answer,
            "ensemble_info": {
                "models_available": len(models) if models else 0,
                "fusion_method": "Weighted Box Fusion"
            }
        }



