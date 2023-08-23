from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from recog import face_detector  # Import the face_detector function from recog.py
app = Flask(__name__)

face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
model = cv2.face_LBPHFaceRecognizer.create()
model.read('trained_model.xml')

def kiosk_mode(frame_generator):
    global locked_status

    while True:
        if locked_status:
            locked_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(locked_image, "Kiosk Locked", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', locked_image)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            yield next(frame_generator)

locked_status = True

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        image, face = face_detector(frame)
        
        if not locked_status: 
            try:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                results = model.predict(face)
                
            except:
                pass

        _, buffer = cv2.imencode('.jpg', image)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if locked_status:
        return Response(kiosk_mode(None), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(kiosk_mode(generate_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status', methods=['GET'])
def get_status():
    global locked_status
    return jsonify({"status": "locked" if locked_status else "unlocked"})

@app.route('/toggle_lock', methods=['POST'])
def toggle_lock():
    global locked_status

    password = request.form.get('password') 

    if password == 'helloworld':
        locked_status = not locked_status  
        return jsonify({"status": "locked" if locked_status else "unlocked"})
    else:
        return jsonify({"error": "Authentication failed"}), 401  

if __name__ == '__main__':
    app.run(debug=True)
