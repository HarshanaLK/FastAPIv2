from fastapi import FastAPI, HTTPException, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from keras.models import load_model

app = FastAPI()

# Add CORS middleware
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained models
emotion_model = load_model('C:/Users/ASUS/Desktop/emotion_model.hdf5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

async def detect_emotion(frame):
    try:
        # Detect faces using Haar Cascade
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return None

        # Assuming only the first face is used for emotion detection
        (x, y, w, h) = faces[0]
        face_roi = frame[y:y + h, x:x + w]
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        face_roi = cv2.resize(face_roi, (64, 64), interpolation=cv2.INTER_AREA)
        face_roi = np.expand_dims(np.expand_dims(face_roi, -1), 0) / 255.0

        # Emotion detection
        emotion_probabilities = emotion_model.predict(face_roi)
        emotion_index = np.argmax(emotion_probabilities)
        detected_emotion = emotion_labels[emotion_index]

        return detected_emotion

    except Exception as e:
        print('Error in emotion detection:', str(e))
        return None

@app.post("/detect_emotion")
async def detect_emotion_route(image: UploadFile = Form(...)):
    try:
        contents = await image.read()
        image_bytes = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail='Failed to decode the image')

        detected_emotion = await detect_emotion(image)

        if detected_emotion:
            print('Detected emotion:', detected_emotion)
            return {'emotion': detected_emotion}
        else:
            print('No face detected or failed to detect emotion')
            raise HTTPException(status_code=400, detail='No face detected or failed to detect emotion')

    except HTTPException as http_exception:
        raise http_exception
    except Exception as e:
        print('Error in /detect_emotion endpoint:', str(e))
        raise HTTPException(status_code=500, detail='Internal Server Error')

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
