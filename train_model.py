import cv2
import os
import numpy as np


# ==========================
# Dataset path
# ==========================

dataset_path = "dataset"


recognizer = cv2.face.LBPHFaceRecognizer_create()


faces = []
ids = []

names = {}

current_id = 0


# ==========================
# Read Dataset
# ==========================

for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(
        dataset_path,
        person_name
    )

    if not os.path.isdir(person_folder):
        continue


    names[current_id] = person_name


    for image_name in os.listdir(person_folder):

        image_path = os.path.join(
            person_folder,
            image_name
        )


        img = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )


        if img is None:
            continue


        faces.append(img)

        ids.append(current_id)


    current_id += 1



print("Images found:", len(faces))


# ==========================
# Train Model
# ==========================

if len(faces) == 0:

    print("No images found!")
    exit()


recognizer.train(
    faces,
    np.array(ids)
)


# ==========================
# Save Model
# ==========================

os.makedirs(
    "trainer",
    exist_ok=True
)


recognizer.save(
    "trainer/trainer.yml"
)



# Save names

with open(
    "trainer/labels.txt",
    "w"
) as f:

    for id,name in names.items():

        f.write(
            f"{id},{name}\n"
        )



print("Training completed successfully")
print("Labels:", names)
