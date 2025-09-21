import os
import cv2
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import mediapipe as mp
from math import cos, sin

# ================================================================
# Load Pretrained Models (Random Forest models for pitch, yaw, roll)
# ================================================================
pitch_model = joblib.load(r"C:\\Users\\Anas\\Models\\rf_pitch_model.pkl")
yaw_model   = joblib.load(r"C:\\Users\\Anas\\Models\\rf_yaw_model.pkl")
roll_model  = joblib.load(r"C:\\Users\\Anas\\Models\\rf_roll_model.pkl")

# ================================================================
# Selected landmark indices (chosen from feature selection step)
# ================================================================
pitch_features_indices = [819, 825, 880, 881, 882, 885, 921, 931, 932, 933]
yaw_features_indices   = [83, 84, 85, 148, 171, 181, 201, 208, 274, 275]
roll_features_indices  = [512, 705, 742, 884, 895, 898, 900, 902, 904, 925]

# ================================================================
# Initialize FastAPI
# ================================================================
app = FastAPI()

# Request schema: input file path
class FilePath(BaseModel):
    path: str

# ================================================================
# File type detection (image vs video)
# ================================================================
def get_file_type(file_path: str) -> str:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    ext = os.path.splitext(file_path)[1].lower()

    if ext in image_exts:
        return "image"
    elif ext in video_exts:
        return "video"
    return "unknown"

# ================================================================
# Mediapipe Face Mesh Setup
# ================================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# ================================================================
# Preprocess landmarks (normalize x, y)
# ================================================================
def preprocess(face, width=450, height=450) -> np.ndarray:
    """
    Normalize face landmarks to be relative to the nose center.
    """
    # Extract x, y coordinates of landmarks
    x_val = [lm.x * width for lm in face.landmark]
    y_val = [lm.y * height for lm in face.landmark]

    # Center landmarks relative to nose tip (landmark 1)
    x_val = np.array(x_val) - np.mean(x_val[1])
    y_val = np.array(y_val) - np.mean(y_val[1])

    # Normalize
    x_val = x_val / x_val.max() if x_val.max() != 0 else x_val
    y_val = y_val / y_val.max() if y_val.max() != 0 else y_val

    return np.concatenate([x_val, y_val])

# ================================================================
# Draw 3D Head Pose Axis on the frame
# ================================================================
def draw_axis(img, pitch, yaw, roll, tdx=None, tdy=None, size=100):
    """
    Draw 3D head pose axes on the image.
    Red = X, Green = Y, Blue = Z
    """
    yaw = -yaw  # Flip yaw direction

    if tdx is None or tdy is None:
        height, width = img.shape[:2]
        tdx, tdy = width // 2, height // 2

    # Compute axis lines
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) +
                 cos(roll) * sin(pitch) * sin(yaw)) + tdy

    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) -
                 sin(pitch) * sin(yaw) * sin(roll)) + tdy

    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    # Draw lines
    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)   # X-axis (Red)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)   # Y-axis (Green)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)   # Z-axis (Blue)

    return img

# ================================================================
# Predict head pose from a single frame
# ================================================================
def predict_single_frame(frame: np.ndarray) -> Dict:
    """
    Predict pitch, yaw, roll from a single video frame or image.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {"pitch": 0, "yaw": 0, "roll": 0}, (w // 2, h // 2)

    face = results.multi_face_landmarks[0]
    marks = preprocess(face, w, h)

    # Extract features for each model
    marks_pitch = marks[pitch_features_indices].reshape(1, -1)
    marks_yaw   = marks[yaw_features_indices].reshape(1, -1)
    marks_roll  = marks[roll_features_indices].reshape(1, -1)

    # Model predictions
    pred_pitch = pitch_model.predict(marks_pitch)[0]
    pred_yaw   = yaw_model.predict(marks_yaw)[0]
    pred_roll  = roll_model.predict(marks_roll)[0]

    # Nose tip = landmark 1
    center = face.landmark[1]
    tdx, tdy = int(center.x * w), int(center.y * h)

    return {"pitch": float(pred_pitch),
            "yaw": float(pred_yaw),
            "roll": float(pred_roll)}, (tdx, tdy)

# ================================================================
# Image Prediction Pipeline
# ================================================================
def predict_image(file_path: str, save_path: str) -> Dict:
    """
    Run head pose prediction on a single image.
    """
    img = cv2.imread(file_path)
    preds, (tdx, tdy) = predict_single_frame(img)

    annotated = draw_axis(img.copy(),
                          preds["pitch"],
                          preds["yaw"],
                          preds["roll"],
                          tdx, tdy, size=100)

    cv2.imwrite(save_path, annotated)
    return preds

# ================================================================
# Video Prediction Pipeline
# ================================================================
def predict_video(file_path: str, save_path: str) -> Dict:
    """
    Run head pose prediction on a video (frame by frame).
    """
    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        save_path, fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    preds_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preds, (tdx, tdy) = predict_single_frame(frame)
        preds_list.append(preds)

        frame = draw_axis(frame, preds["pitch"], preds["yaw"], preds["roll"],
                          tdx, tdy, size=100)
        out.write(frame)

    cap.release()
    out.release()

    return preds_list[-1] if preds_list else {"pitch": 0, "yaw": 0, "roll": 0}

# ================================================================
# FastAPI Endpoint
# ================================================================
@app.post("/predict")
def predict(data: FilePath):
    """
    Endpoint to predict head pose from image or video.
    """
    file_path = data.path
    file_type = get_file_type(file_path)

    if file_type == "unknown":
        return {"error": "Unsupported file type"}

    save_path = os.path.splitext(file_path)[0] + "_predicted" + (
        ".jpg" if file_type == "image" else ".mp4"
    )

    if file_type == "image":
        preds = predict_image(file_path, save_path)
    else:
        preds = predict_video(file_path, save_path)

    return {
        "file_type": file_type,
        "input_path": file_path,
        "output_path": save_path,
        "prediction": preds
    }
# ================================================================
# End of File