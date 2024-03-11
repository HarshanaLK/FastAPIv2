from fastapi import FastAPI, HTTPException, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from keras.models import load_model
from mtcnn.mtcnn import MTCNN

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
mtcnn_detector = MTCNN()
emotion_model = load_model('emotion_model.hdf5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

async def detect_emotion(frame):
    try:
        # Convert the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using MTCNN
        faces = mtcnn_detector.detect_faces(frame)

        if not faces:
            return None

        # Extract keypoints
        keypoints = faces[0]['keypoints']
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']

        # Calculate the center of the eyes
        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

        # Calculate the angle between the eyes
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

        # Get the rotation matrix for rotating the image around the eyes center
        rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

        # Rotate the image to align the eyes horizontally
        aligned_frame = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)

        # Get the face region using MTCNN bounding box
        x, y, width, height = faces[0]['box']
        face_roi = aligned_frame[y:y + height, x:x + width]

        # Resize the face region
        face_roi = cv2.resize(face_roi, (64, 64), interpolation=cv2.INTER_AREA)

        # Preprocess the face for emotion detection
        face_roi = np.expand_dims(np.expand_dims(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY), -1), 0) / 255.0

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