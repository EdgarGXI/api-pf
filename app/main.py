from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
import io
import torch
from ultralytics import YOLO 
import base64

app = FastAPI(title="API for anomaly detection in mammographic images, by Edgar Garcia, Gabriela Bula, Lena Castillo",)

# CORS configuration

origins = [
    "http://localhost",
    "http://localhost:8080", # For testing locally
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, 
    allow_methods=["*"],    
    allow_headers=["*"],    
)


try:
    model = YOLO("best.pt")
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading best.pt: {e}")



@app.get("/")
def raiz():
    return {"message": "API By Edgar Garcia, Gabriela Bula, Lena Castillo"}


@app.post("/predict/")
async def predecir_con_modelo_personalizado(file: UploadFile = File(...)):
    """
    Endpoint for object detection using custom model.
    """
    if not model:
        return {"error": "Model not loaded. Please check the server logs."}

    if not file.content_type.startswith("image/"):
        return {"error": "File type is not an image."}

    contenido_imagen = await file.read()
    try:

        imagen_pil = Image.open(io.BytesIO(contenido_imagen))
    except Exception as e:
        return {"error": f"Image could not be loaded: {e}"}

    try:
       
        resultados_modelo = model(imagen_pil)
    except Exception as e:
        return {"error": f"Error in model inference: {e}"}

    
    api_detections = []
    
    # Create a copy of the image to draw on
    image_to_draw_on = imagen_pil.copy()
    draw_context = ImageDraw.Draw(image_to_draw_on)

    if resultados_modelo: 
        res = resultados_modelo[0]
        class_names_map = res.names 
        
        for box_data in res.boxes:
            class_id = int(box_data.cls)
            confidence_score = round(float(box_data.conf), 4)
            bbox_coordinates = [round(c.item(), 2) for c in box_data.xyxy[0]]
            detected_class_name = class_names_map[class_id]
            
            api_detections.append({
                "class": class_id,          # Expected by Chatbot.jsx for getClassLabel
                "confidence": confidence_score, # Expected by Chatbot.jsx
                "bbox_xyxy": bbox_coordinates,  # For drawing and potential frontend use
                "className": detected_class_name # For drawing the text label
            })

            # Draw bounding box
            draw_context.rectangle(bbox_coordinates, outline="red", width=2)
            
            # Prepare label text
            label_text = f"{detected_class_name} ({confidence_score:.2f})"
            
            # Position text slightly above the bounding box
            text_x_position = bbox_coordinates[0]
            text_y_position = bbox_coordinates[1] - 15  # Adjust offset as needed
            
            # Basic check to keep text within image bounds (if box is at the top)
            if text_y_position < 5: 
                text_y_position = bbox_coordinates[1] + 5 # Place below top edge of box
            
            draw_context.text((text_x_position, text_y_position), label_text, fill="red")


    output_buffer = io.BytesIO()
    image_to_draw_on.save(output_buffer, format="JPEG") # Frontend expects JPEG
    annotated_image_b64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')

    response_payload = {
        "results": api_detections,
        "annotated_image_base64": annotated_image_b64
    }

    if not api_detections:
        response_payload["message"] = "No se detectaron objetos con el modelo personalizado."

    return response_payload
