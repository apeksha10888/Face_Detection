import cv2
import os
import time
from datetime import datetime

# ==========================
# Load Haar Cascade
# ==========================

cascade_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("ERROR: Haar Cascade not found!")
    print("Expected location:")
    print(cascade_path)
    exit()

# ==========================
# Load LBPH Recognizer
# ==========================

recognizer = cv2.face.LBPHFaceRecognizer_create()

model_path = "trainer/trainer.yml"

if not os.path.exists(model_path):
    print("Training model not found!")
    print("Run train_model.py first.")
    exit()

recognizer.read(model_path)

# ==========================
# Load Labels
# ==========================

labels = {}

label_file = "trainer/labels.txt"

if not os.path.exists(label_file):
    print("labels.txt not found!")
    exit()

with open(label_file, "r") as f:
    for line in f:

        line = line.strip()

        if line == "":
            continue

        idx, name = line.split(",", 1)
        labels[int(idx)] = name

# ==========================
# Create folders
# ==========================

os.makedirs("captures", exist_ok=True)

# ==========================
# Open Camera
# ==========================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open camera.")
    exit()

prev = time.time()

# ==========================
# Main Loop
# ==========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame error.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    face_count = len(faces)

    for i, (x, y, w, h) in enumerate(faces):

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        if confidence < 70:

            name = labels.get(label, "Unknown")

            accuracy = max(0, min(100, 100 - confidence))

            color = (0,255,0)

        else:

            name = "Unknown"

            accuracy = max(0, min(100, 100 - confidence))

            color = (0,0,255)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

        cv2.putText(
            frame,
            name,
            (x, y-35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Accuracy: {accuracy:.1f}%",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Face {i+1}",
            (x, y+h+25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    now = time.time()
    fps = 1 / (now - prev)
    prev = now

    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,0),
        2
    )

    cv2.putText(
        frame,
        datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        (10,90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        filename = os.path.join(
            "captures",
            datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        )

        cv2.imwrite(filename, frame)

        print("Saved:", filename)

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
