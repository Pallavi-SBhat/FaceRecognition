import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Load trained LBPH model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

label_dict = {0: "user1", 1: "user2"}   # Change accordingly

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    #convert Frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        # Resize to match training size
        label, confidence = recognizer.predict(face)

        # If confidence is low, face is recognized
        if confidence < 50:
            name = label_dict[label]
            text = f"Access Granted: {name}"
            color = (0,255,0)
        else:
            text = "Access Denied"
            color = (0,0,255)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, text, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()