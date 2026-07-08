import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime


# ==========================
# Theme
# ==========================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ==========================
# Window
# ==========================

app = ctk.CTk()

app.title("AI Face Recognition System")

app.geometry("1280x720")


# ==========================
# Load Haar Cascade
# ==========================

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)


# ==========================
# Load Model
# ==========================

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read(
    "trainer/trainer.yml"
)


# ==========================
# Load Labels
# ==========================

labels = {}

with open("trainer/labels.txt","r") as f:

    for line in f:

        id,name = line.strip().split(",",1)

        labels[int(id)] = name



# ==========================
# Camera Variables
# ==========================

cap = None

camera_running = False



# ==========================
# Camera Processing
# ==========================

def camera_loop():

    global cap
    global camera_running


    previous = time.time()


    while camera_running:


        ret, frame = cap.read()

        if not ret:
            continue


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        faces = face_cascade.detectMultiScale(
            gray,
            1.2,
            5
        )


        app.after(
            0,
            lambda c=len(faces):
            face_count.configure(
                text=f"Faces: {c}"
            )
        )


        for x,y,w,h in faces:


            face = gray[y:y+h,x:x+w]

            face = cv2.resize(
                face,
                (200,200)
            )


            label,confidence = recognizer.predict(face)


            if confidence < 70:

                name = labels.get(
                    label,
                    "Unknown"
                )

                color=(0,255,0)


            else:

                name="Unknown"

                color=(0,0,255)



            app.after(
                0,
                lambda n=name:
                person.configure(
                    text=f"Person: {n}"
                )
            )


            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                color,
                2
            )


            cv2.putText(
                frame,
                name,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )



        now=time.time()

        fps_value=1/(now-previous)

        previous=now


        app.after(
            0,
            lambda f=fps_value:
            fps.configure(
                text=f"FPS: {f:.1f}"
            )
        )



        cv2.putText(
            frame,
            datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            ),
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )



        rgb=cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        img=Image.fromarray(rgb)

        img=img.resize(
            (700,450)
        )


        photo=ImageTk.PhotoImage(img)


        app.after(
            0,
            update_camera,
            photo
        )


        time.sleep(0.03)



def update_camera(photo):

    camera_label.configure(
        image=photo,
        text=""
    )

    camera_label.image=photo




# ==========================
# Camera Control
# ==========================

def start_camera():

    global cap
    global camera_running


    if camera_running:
        return


    cap=cv2.VideoCapture(0)


    if not cap.isOpened():

        status.configure(
            text="Camera Error"
        )

        return


    camera_running=True


    status.configure(
        text="🟢 Camera Connected"
    )


    thread=threading.Thread(
        target=camera_loop,
        daemon=True
    )

    thread.start()



def stop_camera():

    global camera_running
    global cap


    camera_running=False


    if cap:

        cap.release()


    status.configure(
        text="Camera Stopped"
    )


# ==========================
# GUI Layout
# ==========================

title=ctk.CTkLabel(
    app,
    text="🤖 AI FACE RECOGNITION SYSTEM",
    font=("Arial",28,"bold")
)

title.pack(pady=15)



main=ctk.CTkFrame(app)

main.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=10
)



camera_frame=ctk.CTkFrame(main)

camera_frame.pack(
    side="left",
    expand=True,
    fill="both",
    padx=10
)



camera_label=ctk.CTkLabel(
    camera_frame,
    text="Camera Preview",
    font=("Arial",24)
)

camera_label.place(
    relx=.5,
    rely=.5,
    anchor="center"
)



info=ctk.CTkFrame(main)

info.pack(
    side="right",
    fill="y",
    padx=10
)



status=ctk.CTkLabel(
    info,
    text="Camera Stopped",
    font=("Arial",18)
)

status.pack(pady=20)



person=ctk.CTkLabel(
    info,
    text="Person: -",
    font=("Arial",18)
)

person.pack(pady=10)



face_count=ctk.CTkLabel(
    info,
    text="Faces: 0",
    font=("Arial",18)
)

face_count.pack(pady=10)



fps=ctk.CTkLabel(
    info,
    text="FPS: 0",
    font=("Arial",18)
)

fps.pack(pady=10)



buttons=ctk.CTkFrame(app)

buttons.pack(pady=15)



ctk.CTkButton(
    buttons,
    text="▶ Start Camera",
    width=160,
    command=start_camera
).grid(row=0,column=0,padx=10)



ctk.CTkButton(
    buttons,
    text="⏹ Stop",
    width=160,
    command=stop_camera
).grid(row=0,column=1,padx=10)



ctk.CTkButton(
    buttons,
    text="❌ Exit",
    width=160,
    fg_color="red",
    command=app.destroy
).grid(row=0,column=2,padx=10)



# Auto start camera

app.after(
    1000,
    start_camera
)


app.mainloop()
