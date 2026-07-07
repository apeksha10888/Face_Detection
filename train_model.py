import os
import cv2
import numpy as np

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

dataset_path = "dataset"

faces = []
labels = []

label_map = {}
current_label = 0

print("Scanning dataset...\n")

# Read all person folders
for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    label_map[current_label] = person_name

    print(f"Found person: {person_name}")

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        faces.append(image)
        labels.append(current_label)

    current_label += 1

if len(faces) == 0:
    print("No training images found.")
    exit()

print("\nTraining model...")

recognizer.train(faces, np.array(labels))

os.makedirs("trainer", exist_ok=True)

recognizer.save("trainer/trainer.yml")

# Save label map
with open("trainer/labels.txt", "w") as f:
    for label, name in label_map.items():
        f.write(f"{label},{name}\n")

print("\nTraining Complete!")
print(f"People trained: {len(label_map)}")
print(f"Images used: {len(faces)}")
print("Model saved to trainer/trainer.yml")
