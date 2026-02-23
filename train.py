import cv2
import os
import numpy as np

data_path = "dataset"
faces = []
labels = []
label_dict = {}
label_id = 0

for user in os.listdir(data_path):
    label_dict[label_id] = user
    user_path = os.path.join(data_path, user)

    for img in os.listdir(user_path):
        img_path = os.path.join(user_path, img)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 🔥 FIX: Resize all faces to same size
        image = cv2.resize(image, (200, 200))

        faces.append(image)
        labels.append(label_id)

    label_id += 1

faces = np.array(faces)
labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)
recognizer.save("face_model.yml")

print("✅ Training Completed Successfully!")