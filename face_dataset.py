import cv2
import os

face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

name = input("Enter person's name: ")

path = "dataset/" + name

os.makedirs(path, exist_ok=True)


cap = cv2.VideoCapture(0)

count = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )


    for (x,y,w,h) in faces:

        count += 1


        cv2.imwrite(
            f"{path}/{count}.jpg",
            gray[y:y+h, x:x+w]
        )


        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (255,0,0),
            2
        )


    cv2.putText(
        frame,
        f"Images: {count}/100",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )


    cv2.imshow(
        "Dataset Collection",
        frame
    )


    if cv2.waitKey(1) == ord("q"):
        break


    if count >= 100:
        break


print("Dataset completed")


cap.release()
cv2.destroyAllWindows()
