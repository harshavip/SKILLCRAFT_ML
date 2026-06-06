import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Dataset folders
cat_folder = "dataset/cats"
dog_folder = "dataset/dogs"

# Image size
image_size = 64

data = []
labels = []


# -------------------------------
# Load cat images
# -------------------------------
for image_name in os.listdir(cat_folder):
    image_path = os.path.join(cat_folder, image_name)
    image = cv2.imread(image_path)

    if image is not None:
        image = cv2.resize(image, (image_size, image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.flatten()

        data.append(image)
        labels.append(0)  # 0 = Cat


# -------------------------------
# Load dog images
# -------------------------------
for image_name in os.listdir(dog_folder):
    image_path = os.path.join(dog_folder, image_name)
    image = cv2.imread(image_path)

    if image is not None:
        image = cv2.resize(image, (image_size, image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.flatten()

        data.append(image)
        labels.append(1)  # 1 = Dog


# -------------------------------
# Convert to numpy arrays
# -------------------------------
X = np.array(data)
y = np.array(labels)

print("Total images loaded:", len(X))
print("Cat images:", list(y).count(0))
print("Dog images:", list(y).count(1))
print("Image feature size:", X.shape)


# -------------------------------
# Check dataset
# -------------------------------
if len(X) == 0:
    print("No images found. Check dataset/cats and dataset/dogs folders.")
    exit()

if list(y).count(0) == 0 or list(y).count(1) == 0:
    print("Both cat and dog images are required.")
    exit()


# -------------------------------
# Normalize pixel values
# -------------------------------
X = X / 255.0


# -------------------------------
# Split dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# -------------------------------
# Create SVM model
# -------------------------------
model = SVC(kernel="linear")


# -------------------------------
# Train model
# -------------------------------
print("\nTraining SVM model...")
model.fit(X_train, y_train)


# -------------------------------
# Predict test data
# -------------------------------
y_pred = model.predict(X_test)


# -------------------------------
# Evaluation
# -------------------------------
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Cat", "Dog"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -------------------------------
# Function to predict selected image
# -------------------------------
def predict_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found or invalid image.")
        return

    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess image
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.flatten()
    image = image / 255.0
    image = image.reshape(1, -1)

    # Predict
    prediction = model.predict(image)

    if prediction[0] == 0:
        result = "Cat"
    else:
        result = "Dog"

    print("\nSelected Image:", image_path)
    print("Prediction:", result)

    # Show image with prediction
    plt.imshow(original_image)
    plt.title("Predicted: " + result)
    plt.axis("off")
    plt.show()


# -------------------------------
# Get all images from dataset folder only
# -------------------------------
image_files = []

# Add cat images
for file in os.listdir(cat_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        image_files.append(os.path.join(cat_folder, file))

# Add dog images
for file in os.listdir(dog_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        image_files.append(os.path.join(dog_folder, file))


if len(image_files) == 0:
    print("\nNo images found in dataset folder.")
    exit()


# -------------------------------
# Ask repeatedly until user says no
# -------------------------------
while True:
    print("\nAvailable Images from Dataset:")

    for index, file_path in enumerate(image_files):
        print(f"{index + 1}. {file_path}")

    try:
        choice = int(input("\nEnter image number to predict: "))

        if choice < 1 or choice > len(image_files):
            print("Invalid choice. Please enter a valid number.")
        else:
            selected_path = image_files[choice - 1]
            predict_image(selected_path)

    except ValueError:
        print("Please enter only a number.")

    again = input("\nDo you want to predict another image? (yes/no): ").lower()

    if again != "yes":
        print("Program ended.")
        break