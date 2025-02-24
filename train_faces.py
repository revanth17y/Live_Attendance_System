import face_recognition as fr
import os
import pickle
import sqlite3

def test():
    ENCODINGS_FILE = "encodings.pickle"
    KNOWN_FACES_DIR = "known_faces"

    # Load existing encodings
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        knownEncodings = data["encodings"]
        names = data["names"]
    except (FileNotFoundError, EOFError):
        knownEncodings = []
        names = []

    # Connect to SQLite
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY, name TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY, name TEXT, timestamp TEXT)")
    # Process images
    new_images_trained = False

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            name = os.path.splitext(filename)[0]

            if name in names:
                continue  # Skip already trained images

            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = fr.load_image_file(image_path)
            face_locations = fr.face_locations(image)

            if face_locations:
                encoding = fr.face_encodings(image)[0]
                knownEncodings.append(encoding)
                names.append(name)

                cursor.execute("INSERT OR IGNORE INTO students (name) VALUES (?)", (name,))
                new_images_trained = True
            else:
                print(f"⚠ No face detected in {filename}, skipping...")

    # Save updated encodings only if new faces were added
    if new_images_trained:
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump({"encodings": knownEncodings, "names": names}, f)
        print("✅ New face trained successfully!")
    else:
        print("⚠ No new faces were trained.")

    conn.commit()
    conn.close()
test()