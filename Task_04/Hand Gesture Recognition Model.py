import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dataset_path = r"C:\Users\Vignesh\Downloads\LeapGestRecog"

X = []
Y = []

label_dict = {}
label = 0

for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if os.path.isdir(person_path):

        for gesture in os.listdir(person_path):

            gesture_path = os.path.join(person_path, gesture)

            if gesture not in label_dict:
                label_dict[gesture] = label
                label += 1

            for image_name in os.listdir(gesture_path):

                image_path = os.path.join(gesture_path, image_name)

                try:
                    img = cv2.imread(image_path, 0)

                    img = cv2.resize(img, (64, 64))

                    img = img.flatten()

                    X.append(img)

                    Y.append(label_dict[gesture])

                except:
                    pass

X = np.array(X)
Y = np.array(Y)

print("Total Images:", len(X))

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

model = SVC(kernel='linear')

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)

print("Accuracy:", accuracy * 100)

print("\nGesture Labels")
for key, value in label_dict.items():
    print(value, "=", key)

# Predict New Gesture

test_image = r"C:\Users\Vignesh\Downloads\gesture.jpg"

img = cv2.imread(test_image, 0)

img = cv2.resize(img, (64, 64))

img = img.flatten()

prediction = model.predict([img])

for gesture, label in label_dict.items():
    if label == prediction[0]:
        print("\nPredicted Gesture:", gesture)
        break
