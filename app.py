from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from recog import face_detector  # Import the face_detector function from recog.py

app = Flask(__name__)

face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
model = cv2.face_LBPHFaceRecognizer.create()
model.read('trained_model.xml')

locked_status = True  # Initialize as locked by default

def generate_frames():
    cap = cv2.VideoCapture(0)
    global locked_status  # Use the global variable
    
    while True:
        ret, frame = cap.read()
        image, face = face_detector(frame)  # Use the imported face_detector function
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            results = model.predict(face)
            
            if results[1] < 500:
                confidence = int(100 * (1 - results[1] / 400))
                display_string = f"{confidence}% Confident it is User"
                locked_status = False  # Unlock when face is detected
            else:
                confidence = int(100 * (1 - results[1] / 400))
                display_string = f"{confidence}% Not Recognized"
                locked_status = True  # Lock when face is not detected
            
            cv2.putText(image, display_string, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 120, 150), 2)

            if confidence > 40:
                cv2.putText(image, "Unlocked", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(image, "Locked", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

            _, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except:
            cv2.putText(image, "No Face Found", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, "Locked", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            locked_status = True  # Lock when no face is found

            _, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status', methods=['GET'])
def get_status():
    global locked_status
    return jsonify({"status": "locked" if locked_status else "unlocked"})



# New API endpoint
@app.route('/api/data', methods=['GET'])
def get_data():
    data = {
        "key1": "value1",
        "key2": "value2"
    }
    return jsonify(data)



if __name__ == '__main__':
    app.run(debug=True)
