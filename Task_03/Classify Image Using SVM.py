import os
import cv2
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = []
Y = []

cat_path = r"C:\Users\THATI VIGNESH\Downloads\archive.zip\PetImages\Cat"
dog_path = r"C:\Users\THATI VIGNESH\Downloads\archive.zip\PetImages\Dog"

# Load Cat Images
for image_name in os.listdir(cat_path):
    image_path = os.path.join(cat_path, image_name)

    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (64, 64))
        img = img.flatten()

        X.append(img)
        Y.append(0)  # Cat

    except:
        pass

# Load Dog Images
for image_name in os.listdir(dog_path):
    image_path = os.path.join(dog_path, image_name)

    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (64, 64))
        img = img.flatten()

        X.append(img)
        Y.append(1)  # Dog

    except:
        pass

X = np.array(X)
Y = np.array(Y)

print("Total Images:", len(X))

# Split Dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

# Train SVM
model = SVC(kernel='linear')
model.fit(X_train, Y_train)

# Predict
Y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(Y_test, Y_pred)

print("Accuracy:", accuracy * 100, "%")

# Test New Image
test_image = r"C:\Users\THATI VINESH\Downloads\test.jpg"

img = cv2.imread(test_image)
img = cv2.resize(img, (64, 64))
img = img.flatten()

prediction = model.predict([img])

if prediction[0] == 0:
    print("Cat")
else:
    print("Dog")
