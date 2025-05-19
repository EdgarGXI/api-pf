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
app = FastAPI(
    title="API for anomaly detection in mammographic images, by Edgar Garcia, Gabriela Bula, Lena Castillo"
)

# CORS
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
    "*",
]
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
            photometric_interpretation = getattr(
                dicom, "PhotometricInterpretation", "MONOCHROME2"
            )

        # Transforms raw DICOM data to "human-friendly" view
        data = (
            apply_voi_lut(dicom.pixel_array, dicom)
            if self.voi_lut
            else dicom.pixel_array
        )

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
    pil_image = Image.fromarray(processed_array, mode="L")  # 'L' for grayscale

    # Convert to RGB if needed for the model
    pil_image = pil_image.convert("RGB")

    return pil_image


def ensemble_predict_wbf_single_image(
    models, image, iou_threshold=0.5, confidence_threshold=0.25, skip_box_thr=0.0001
):
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
                relative_boxes = boxes / np.array(
                    [img_width, img_height, img_width, img_height]
                )

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
            all_boxes,
            all_scores,
            all_classes,
            iou_thr=iou_threshold,
            skip_box_thr=skip_box_thr,
        )

        # Convert back to absolute coordinates if we have detections
        if len(fused_boxes) > 0 and len(individual_results) > 0:
            img_height, img_width = individual_results[0].orig_shape[:2]
            fused_boxes = fused_boxes * np.array(
                [img_width, img_height, img_width, img_height]
            )

        # Create detection results in the expected format
        detections = []
        for i, (box, score, cls) in enumerate(
            zip(fused_boxes, fused_scores, fused_classes)
        ):
            detections.append(
                {
                    "class": int(cls),
                    "confidence": round(float(score), 4),
                    "bbox_xyxy": [round(float(coord), 2) for coord in box],
                    "className": "anomaly" if int(cls) == 0 else "normal",
                }
            )

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

                    detections.append(
                        {
                            "class": cls,
                            "confidence": conf,
                            "bbox_xyxy": bbox,
                            "className": name,
                        }
                    )
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
        if file.filename.lower().endswith(".dicom") or file.filename.lower().endswith(
            ".dcm"
        ):
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
            detections, class_label, individual_results = (
                ensemble_predict_wbf_single_image(models, image)
            )
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
        annotated_image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        if detections:
            max_confidence = max([d["confidence"] for d in detections])
            confidence_str = f"{max_confidence:.2f}"
        else:
            confidence_str = "N/A"

        # Interpret detection results
        if detections:
            if class_label == 0:
                detection_result = f"Anomalies were detected in the mammographic image"
            else:
                detection_result = (
                    f"No anomalies were detected in the mammographic image"
                )
        else:
            detection_result = f"No objects were detected in the image"

        # Generate contextual explanation with OpenAI
        try:

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an expert in explaining mammography and medical imaging results, with a deep understanding of machine learning, "
                            "specifically concerning the diagnosis of breast cancer. You are familiar with a final project from Universidad del Norte titled "
                            "'Modelo de inteligencia artificial para el diagnóstico asistido del cáncer de mama: Un enfoque basado en redes neuronales'. "
                            "This project aimed to develop and implement an AI application based on the YOLO (You Only Look Once) architecture for the early "
                            "detection of breast cancer. The justification for the project includes the high global prevalence and mortality of breast cancer, "
                            "the significant percentage of misclassified mammograms due to human factors (interpretation and image search errors), and deficiencies "
                            "in mammography quality in regions like Colombia.\n\n"
                            "The AI model specifically uses the YOLOv8 architecture, chosen for its real-time object detection capabilities with Convolutional Neural Networks (CNNs). "
                            "YOLOv8's architecture consists of an input layer (processing 640x640x3 pixel images), a Backbone (CSPDarknet53, capturing multi-scale features using "
                            "kernels, strides, and padding, notably 3x3 kernels and stride 2 for progressive feature map reduction, and incorporating C2f blocks and an SPPF module), "
                            "a Neck (using PANet for multi-scale information flow optimization and replacing FPN with C2f modules), and a Head (generating bounding boxes "
                            "and confidence scores). The detection block uses 4xreg_max parameterization for bounding box regression and a classification module, "
                            "optimized with CIoU and Distribution Focal Loss (DFL).\n\n"
                            "The project utilized the VinDr-Mammo dataset, a large-scale public dataset of 5,000 four-view digital mammography exams with BI-RADS assessments and "
                            "finding annotations (masses, calcifications, nodes), including breast-level_annotations.csv and finding_annotations.csv. "
                            "From this, 1659 cases were selected (1363 with masses/calcifications BI-RADS > 3, and 296 normal cases) to create a less imbalanced dataset (85% anomalous, 15% normal).\n\n"
                            "Methodology involved several stages: \n"
                            "1. Data Exploration: Review of literature and selection of VinDr-Mammo.\n"
                            "2. Preprocessing: Conversion of DICOM files to PNG. Application of advanced image processing techniques including segmentation with CLAHE filters to improve contrast, "
                            "standardization of intensities, correction of negative coordinates in bounding boxes (to 0) and those exceeding image dimensions (to max allowed), "
                            "and conversion of bounding box coordinates to YOLO format (normalized x_center, y_center, width, height). Images were resized to 640x640 pixels and "
                            "converted from MONOCHROME1 to MONOCHROME2.\n"
                            "3. Data Augmentation: Generation of three augmented versions per image using horizontal flips, vertical flips, and 90-degree rotations with Albumentations.\n"
                            "4. Hyperparameter Tuning (Tuning): Initial reduced training (20 epochs, 50 iterations) using Optuna to estimate hyperparameters like initial learning rate (lr0), "
                            "final learning rate factor (lrf), SGD momentum, L2 regularization (weight_decay), warmup epochs, warmup momentum, bounding box loss weight (box), and classification loss weight (cls).\n"
                            "5. Model Training and Validation: The YOLOv8-L model was trained for 128 epochs using PyTorch and an NVIDIA RTX A5000 GPU. "
                            "The dataset was split into training (70%), validation (15%), and testing (15%) using stratified division based on laterality, view_position, breast_density, and finding_categories.\n\n"
                            "Key results from the validation included a confusion matrix with 783 True Positives (VP), 325 True Negatives (VN), 348 False Positives (FP), and 29 False Negatives (FN). "
                            "Performance metrics showed: \n"
                            "- Precision: 96% for anomalies (class 0), 48% for normal tissue (class 1).\n"
                            "- Sensitivity (Recall): 73% for anomalies (class 0), 92% for normal tissue (class 1).\n"
                            "- F1-Score: 83% for anomalies (class 0), 60% for normal tissue (class 1).\n"
                            "- Overall Accuracy: 74%.\n"
                            "- Localization Performance: mAP50 (mean Average Precision at IoU 0.5) of 52%, and mAP50-95 (mAP averaged over IoU thresholds from 0.5 to 0.95) of 31%.\n\n"
                            "The project also involved developing a prototype with a React frontend (deployed via GitHub Pages). "
                            "The user interface included a Chatbot for queries and a Dashboard for visualizing metrics.\n\n"
                            "You should be prepared to answer questions about these project specifics, the general concepts of Deep Learning (multi-layered neural networks emulating human decision-making), "
                            "Convolutional Neural Networks (specialized for image processing using matrix multiplication), Mammography (X-ray of the breast for screening, detecting tumors and microcalcifications), "
                            "Neoplasia (uncontrolled growth of abnormal cells), different YOLOv8 versions (n, s, m, l, x), and the role of AI in improving diagnostic accuracy and speed in medical imaging, "
                            "particularly in breast cancer detection by identifying and localizing suspicious areas with bounding boxes and confidence scores, and understanding the BI-RADS classification system. "
                            "Acknowledge that AI tools are for decision support and do not replace specialized medical judgment."
                        ,
                    },
                    {
                        "role": "user",
                        "content": f"Regarding: {detection_result} Based on this result with this confidence {confidence_str}, please answer: {question}",
                    },
                ],
                max_tokens=500,
                temperature=0.2,
            )
            explanation = response.choices[0].message.content
        except Exception as e:
            explanation = f"Error generating explanation: {e}"

        return {
            "results": detections,
            "annotated_image_base64": annotated_image_b64,
            "detection_result": detection_result,
            "explanation": explanation,
            "file_type": (
                "dicom"
                if file.filename.lower().endswith((".dicom", ".dcm"))
                else "image"
            ),
            "ensemble_info": {
                "models_used": len(models),
                "model_paths": [
                    path.split("/")[-1]
                    for path in ["app/best.pt", "app/best2.pt", "app/best3.pt"]
                    if models
                ],
                "fusion_method": "Weighted Box Fusion",
            },
        }

    # If no image uploaded, answer general question
    else:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in explaining mammography and medical imaging results, with a deep understanding of machine learning, "
                            "specifically concerning the diagnosis of breast cancer. You are familiar with a final project from Universidad del Norte titled "
                            "'Modelo de inteligencia artificial para el diagnóstico asistido del cáncer de mama: Un enfoque basado en redes neuronales'. "
                            "This project aimed to develop and implement an AI application based on the YOLO (You Only Look Once) architecture for the early "
                            "detection of breast cancer. The justification for the project includes the high global prevalence and mortality of breast cancer, "
                            "the significant percentage of misclassified mammograms due to human factors (interpretation and image search errors), and deficiencies "
                            "in mammography quality in regions like Colombia.\n\n"
                            "The AI model specifically uses the YOLOv8 architecture, chosen for its real-time object detection capabilities with Convolutional Neural Networks (CNNs). "
                            "YOLOv8's architecture consists of an input layer (processing 640x640x3 pixel images), a Backbone (CSPDarknet53, capturing multi-scale features using "
                            "kernels, strides, and padding, notably 3x3 kernels and stride 2 for progressive feature map reduction, and incorporating C2f blocks and an SPPF module), "
                            "a Neck (using PANet for multi-scale information flow optimization and replacing FPN with C2f modules), and a Head (generating bounding boxes "
                            "and confidence scores). The detection block uses 4xreg_max parameterization for bounding box regression and a classification module, "
                            "optimized with CIoU and Distribution Focal Loss (DFL).\n\n"
                            "The project utilized the VinDr-Mammo dataset, a large-scale public dataset of 5,000 four-view digital mammography exams with BI-RADS assessments and "
                            "finding annotations (masses, calcifications, nodes), including breast-level_annotations.csv and finding_annotations.csv. "
                            "From this, 1659 cases were selected (1363 with masses/calcifications BI-RADS > 3, and 296 normal cases) to create a less imbalanced dataset (85% anomalous, 15% normal).\n\n"
                            "Methodology involved several stages: \n"
                            "1. Data Exploration: Review of literature and selection of VinDr-Mammo.\n"
                            "2. Preprocessing: Conversion of DICOM files to PNG. Application of advanced image processing techniques including segmentation with CLAHE filters to improve contrast, "
                            "standardization of intensities, correction of negative coordinates in bounding boxes (to 0) and those exceeding image dimensions (to max allowed), "
                            "and conversion of bounding box coordinates to YOLO format (normalized x_center, y_center, width, height). Images were resized to 640x640 pixels and "
                            "converted from MONOCHROME1 to MONOCHROME2.\n"
                            "3. Data Augmentation: Generation of three augmented versions per image using horizontal flips, vertical flips, and 90-degree rotations with Albumentations.\n"
                            "4. Hyperparameter Tuning (Tuning): Initial reduced training (20 epochs, 50 iterations) using Optuna to estimate hyperparameters like initial learning rate (lr0), "
                            "final learning rate factor (lrf), SGD momentum, L2 regularization (weight_decay), warmup epochs, warmup momentum, bounding box loss weight (box), and classification loss weight (cls).\n"
                            "5. Model Training and Validation: The YOLOv8-L model was trained for 128 epochs using PyTorch and an NVIDIA RTX A5000 GPU. "
                            "The dataset was split into training (70%), validation (15%), and testing (15%) using stratified division based on laterality, view_position, breast_density, and finding_categories.\n\n"
                            "Key results from the validation included a confusion matrix with 783 True Positives (VP), 325 True Negatives (VN), 348 False Positives (FP), and 29 False Negatives (FN). "
                            "Performance metrics showed: \n"
                            "- Precision: 96% for anomalies (class 0), 48% for normal tissue (class 1).\n"
                            "- Sensitivity (Recall): 69% for anomalies (class 0), 92% for normal tissue (class 1).\n"
                            "- F1-Score: 79% for anomalies (class 0), 60% for normal tissue (class 1).\n"
                            "- Overall Accuracy: 72%.\n"
                            "- Localization Performance: mAP50 (mean Average Precision at IoU 0.5) of 52%, and mAP50-95 (mAP averaged over IoU thresholds from 0.5 to 0.95) of 31%.\n\n"
                            "The project also involved developing a prototype with a React frontend (deployed via GitHub Pages). "
                            "The user interface included a Chatbot for queries and a Dashboard for visualizing metrics.\n\n"
                            "You should be prepared to answer questions about these project specifics, the general concepts of Deep Learning (multi-layered neural networks emulating human decision-making), "
                            "Convolutional Neural Networks (specialized for image processing using matrix multiplication), Mammography (X-ray of the breast for screening, detecting tumors and microcalcifications), "
                            "Neoplasia (uncontrolled growth of abnormal cells), different YOLOv8 versions (n, s, m, l, x), and the role of AI in improving diagnostic accuracy and speed in medical imaging, "
                            "particularly in breast cancer detection by identifying and localizing suspicious areas with bounding boxes and confidence scores, and understanding the BI-RADS classification system. "
                            "Acknowledge that AI tools are for decision support and do not replace specialized medical judgment.",
                    },
                    {"role": "user", "content": question},
                ],
                max_tokens=500,
                temperature=0.7,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error generating response from OpenAI: {e}"

        return {
            "response": answer,
            "ensemble_info": {
                "models_available": len(models) if models else 0,
                "fusion_method": "Weighted Box Fusion",
            },
        }
