# Head Pose Estimation API

This project provides a **Head Pose Estimation pipeline** using: -
**Mediapipe Face Mesh** (for facial landmarks detection)\
- **Random Forest models** (trained for pitch, yaw, and roll
prediction)\
- **FastAPI** (for serving predictions as an API)\
- **OpenCV** (for visualization and processing images/videos)

The system predicts **pitch, yaw, and roll** angles from images or
videos, and draws the 3D axis on the detected face.

------------------------------------------------------------------------

## 📂 Project Structure

    .
    ├── models/                      # Trained Random Forest models
    │   ├── rf_pitch_model.pkl
    │   ├── rf_yaw_model.pkl
    │   └── rf_roll_model.pkl
    ├── api.py                       # FastAPI server with prediction endpoints
    ├── notebook.ipynb               # Colab notebook (data preprocessing + training)
    └── README.md                    # Documentation

------------------------------------------------------------------------

## ⚙️ Features

-   Supports both **images** and **videos**
-   Runs **face landmark detection** using Mediapipe
-   Extracts selected landmarks for prediction
-   Predicts head pose angles using **Random Forest models**
-   Annotates the image/video with **3D pose axes**
-   Exposes a **FastAPI endpoint** for easy integration

------------------------------------------------------------------------

## 📊 Models

The models were trained in a **Google Colab notebook** using: -
**Dataset:**
[AFLW2000-3D](https://wywu.github.io/projects/AFLW2000/AFLW2000.html)\
- **Algorithm:** Random Forest (scikit-learn)\
- **Features:** Selected facial landmarks indices for each angle (pitch,
yaw, roll)

Trained models are saved as `.pkl` files: - `rf_pitch_model.pkl` -
`rf_yaw_model.pkl` - `rf_roll_model.pkl`

------------------------------------------------------------------------

## 🚀 API Usage

### 1. Install dependencies

``` bash
pip install fastapi uvicorn opencv-python mediapipe scikit-learn joblib
```

### 2. Run FastAPI server

``` bash
uvicorn api:app --reload
```

### 3. Send a request

POST request with the file path:

``` json
{
  "path": "C:/Users/Anas/test_image.jpg"
}
```

### 4. Example Response

``` json
{
  "file_type": "image",
  "input_path": "C:/Users/Anas/test_image.jpg",
  "output_path": "C:/Users/Anas/test_image_predicted.jpg",
  "prediction": {
    "pitch": -8.21,
    "yaw": 3.44,
    "roll": 1.09
  }
}
```

For videos, the response will include the processed video path.

------------------------------------------------------------------------

## 🖼️ Visualization

-   Red axis → **X direction**\
-   Green axis → **Y direction**\
-   Blue axis → **Z direction**

------------------------------------------------------------------------

## 📒 Colab Notebook

The Colab notebook includes: - Loading dataset (`AFLW2000-3D`) -
Extracting facial landmarks - Preprocessing & normalization - Feature
selection - Training Random Forest models - Saving `.pkl` models for API
usage

------------------------------------------------------------------------

## 🔧 Requirements

-   Python 3.8+
-   FastAPI
-   Uvicorn
-   OpenCV
-   Mediapipe
-   Scikit-learn
-   Joblib
-   NumPy

Install everything with:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📌 Notes

-   Make sure the models (`.pkl`) are placed in the correct path (update
    in `api.py` if needed).\
-   Works best with **frontal face images/videos**.\
-   Default supported formats:
    -   Images: `.jpg, .jpeg, .png, .bmp`
    -   Videos: `.mp4, .avi, .mov, .mkv`

------------------------------------------------------------------------

## 📝 License

This project is open-source and available under the MIT License.
