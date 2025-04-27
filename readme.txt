Facial Recognition-Based Attendance System

Overview
This is a Flask-powered attendance system that uses real-time facial recognition to track and record attendance. It captures faces from a live video feed, matches them with stored encodings, and logs attendance in an SQLite database. The system provides a web-based interface for viewing and exporting attendance records.

Features
- Real-time face recognition for attendance tracking.
- Stores attendance records in an SQLite database.
- Web-based interface using Flask.
- Ability to add new faces dynamically.
- Automatic CSV generation for attendance reports.
- Remote accessibility if hosted on a server.

 Requirements
Ensure you have Python installed, then install the required dependencies:

sh

pip install -r requirements.txt



### Dependencies
- Flask
- OpenCV (`opencv-python`)
- Face Recognition (`face-recognition`)
- NumPy
- Pillow
- SQLite3 (built into Python)

### Install CMake and Visual Studio Build Tools to install face_recognition:
- CMake: [Download](https://cmake.org/download/)
- Visual Studio Build Tools: [Download](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

## Setup Instructions

### 1. Prepare the Environment
Run the following commands:
```sh

python.exe -m pip install --upgrade pip

python -m venv env

env\Scripts\activate

pip install -r requirements.txt

```

### 2. Train the Face Recognition Model
Before running the application, execute `train_faces.py` at least once to create the required database and face encodings:
```sh

python train_faces.py

```

To train the faces to identify, ensure you paste your images into the `known_faces` folder and save them as `.png`, `.jpeg`, or `.jpg`.

### 3. Run the Application
Start the Flask server:
```sh

python app.py

```
Then, open `http://127.0.0.1:5000/` in your web browser.

## Usage
- The system captures a live video feed and detects faces.
- Recognized faces are marked as present and stored in the database.
- The attendance table updates in real-time without refreshing the page.
- Attendance records can be exported as a CSV file.
- New faces can be added using the `/new_face` interface.

## Folder Structure
```
face_attendance_system/
│── known_faces/          # Store images of known faces
│── database.db           # SQLite database for attendance records
│── encodings.pickle      # Serialized face encodings
│── app.py                # Main Flask application
│── train_faces.py        # Face encoding generator
│── templates/            # HTML files for Flask
│── static/               # CSS and JavaScript files
│── requirements.txt      # Required dependencies
│── README.md             # Project documentation
```

## API Endpoints
| Endpoint                | Method | Description                                    |
|-------------------------|--------|------------------------------------------------|
| `/`                     | GET    | Displays the main page with attendance records |
| `/video_feed`           | GET    | Provides the live video feed                   |
| `/get_attendance`       | GET    | Returns attendance records in JSON format      |
| `/add_face`             | POST   | Adds a new face to the database                |
| `/download_attendance`  | GET    | Downloads attendance records as a CSV file     |


## Author
Developed by YARAM REVANTH KUMAR
