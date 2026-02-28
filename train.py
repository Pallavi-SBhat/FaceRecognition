import cv2
import os
import numpy as np

# Path to dataset folder
data_path = "dataset"
faces = []      # List to store face images
labels = []     # List to store corresponding labels
label_dict = {} # Dictionary to map label numbers to names
label_id = 0    # Numerical label counter

# Loop through each user folder inside dataset
for user in os.listdir(data_path):
    label_dict[label_id] = user
    user_path = os.path.join(data_path, user)

    for img in os.listdir(user_path):
        img_path = os.path.join(user_path, img)

        # Read image in grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        #  FIX: Resize all faces to same size
        image = cv2.resize(image, (200, 200))

        faces.append(image)
        labels.append(label_id)

    label_id += 1

faces = np.array(faces)
labels = np.array(labels)

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train recognizer with face data and labels
recognizer.train(faces, labels)

# Save trained model
recognizer.save("face_model.yml")

print(" Training Completed Successfully!")