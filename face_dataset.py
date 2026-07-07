import cv2
import os

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Ask for person's name
person_name = input("Enter person's name: ").strip()

if person_name == "":
    print("Invalid name!")
    exit()

# Create folder
dataset_path = os.path.join("dataset", person_name)
os.makedirs(dataset_path, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0
max_images = 100

print("\nLook at the camera.")
print("Move your face slightly left, right, up, and down.")
print("Capturing images...\n")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        count += 1

        filename = os.path.join(dataset_path, f"{count}.jpg")
        cv2.imwrite(filename, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(
            frame,
            f"Images: {count}/{max_images}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        cv2.imshow("Face Dataset Collection", frame)

        cv2.waitKey(100)

        if count >= max_images:
            break

    cv2.imshow("Face Dataset Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nDataset completed for {person_name}")
print(f"Images saved: {count}")
