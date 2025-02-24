from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import face_recognition as fr
import pickle
import numpy as np
import sqlite3
import os
from datetime import datetime
import subprocess
import time
from werkzeug.utils import secure_filename
from train_faces import test  # Custom training function

# Initialize Flask app
app = Flask(__name__)

# Configuration
ENCODINGS_FILE = "encodings.pickle"
UPLOAD_FOLDER = "known_faces"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Camera control flag
camera_active = True

# Initialize encodings
def load_encodings():
    """Loads face encodings from pickle file."""
    global knownEncodings, names
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        knownEncodings = data.get("encodings", [])
        names = data.get("names", [])
        print("✅ Encodings loaded successfully.")
    except (FileNotFoundError, EOFError):
        knownEncodings = []
        names = []
        print(f"❌ Error: '{ENCODINGS_FILE}' not found or empty.")

load_encodings()

# Clear attendance on startup
def clear_attendance():
    """Deletes all attendance records on application restart."""
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance")
    conn.commit()
    conn.close()

clear_attendance()

# Initialize camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Error: Unable to access the camera.")
    exit(1)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ------------------------------
# FACE RECOGNITION FUNCTIONALITY
# ------------------------------
def recognize_faces():
    """Performs real-time face recognition and updates attendance."""
    global camera_active
    while True:
        if not camera_active:
            time.sleep(1)  # Pause if camera is inactive
            continue

        success, frame = cam.read()
        if not success:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        facelocations = fr.face_locations(rgb_frame)
        encodings = fr.face_encodings(rgb_frame, facelocations)

        try:
            conn = sqlite3.connect("database.db")
            cursor = conn.cursor()

            for face_location, encoding in zip(facelocations, encodings):
                top, right, bottom, left = face_location
                name = "Unknown"

                if knownEncodings:
                    distances = fr.face_distance(knownEncodings, encoding)
                    best_match_index = np.argmin(distances)

                    if distances[best_match_index] < 0.6:
                        name = names[best_match_index]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                if name != "Unknown":
                    timestamp_now = datetime.now()
                    cursor.execute("SELECT * FROM attendance WHERE name = ?", (name,))
                    existing_record = cursor.fetchone()

                    if existing_record:
                        cursor.execute("UPDATE attendance SET timestamp = ? WHERE name = ?", (timestamp_now, name))
                    else:
                        cursor.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, timestamp_now))

                    conn.commit()
        except sqlite3.Error as e:
            print(f"❌ Database error: {e}")
        finally:
            conn.close()

        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ------------------------------
# ROUTES
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Streams video feed if camera is active."""
    global camera_active
    if camera_active:
        return Response(recognize_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response("", mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    global cam
    if cam.isOpened():
        cam.release()
    return jsonify({"status": "Camera stopped"})


@app.route('/start_camera')
def start_camera():
    """Starts the camera feed."""
    global camera_active, cam
    camera_active = True

    # Reinitialize camera if it's closed
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            return jsonify({'status': '❌ Error: Unable to access the camera'})

    return jsonify({'status': '✅ Camera started'})


@app.route('/attendance')
def attendance():
    """Fetches attendance records and stops the camera before rendering the page."""
    global camera_active
    camera_active = False  # Stop camera when viewing attendance

    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance ORDER BY timestamp DESC")
        data = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        data = []
    finally:
        conn.close()

    return render_template('attendance.html', records=data)

# ------------------------------
# IMAGE UPLOAD & TRAINING
# ------------------------------
def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_face', methods=['POST'])
def upload_face():
    """Uploads a new image, trains the model, and reloads encodings."""
    if "image" not in request.files or "name" not in request.form:
        return "❌ Error: Missing file or name!", 400

    file = request.files["image"]
    name = request.form["name"].strip()

    if file.filename == "" or not allowed_file(file.filename):
        return "❌ Error: Invalid file!", 400

    filename = secure_filename(name + ".jpg")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    subprocess.run(["python", "train_faces.py"], check=True)

    time.sleep(2)
    load_encodings()
    test()

    return redirect(url_for('index'))

# ------------------------------
# FETCH RECENT ATTENDANCE
# ------------------------------
@app.route('/recent_attendance')
def recent_attendance():
    """Fetches the most recent attendance records (last 10 entries)."""
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name, timestamp FROM attendance ORDER BY timestamp DESC LIMIT 10")
        records = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        records = []
    finally:
        conn.close()

    attendance_data = []
    for name, timestamp in records:
        dt_obj = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
        attendance_data.append({
            "name": name,
            "date": dt_obj.strftime("%Y-%m-%d"),
            "time": dt_obj.strftime("%H:%M:%S")
        })

    return jsonify({"attendance": attendance_data})

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
