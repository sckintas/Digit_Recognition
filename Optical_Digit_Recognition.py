import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Normalize data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Convert target vectors to categorical targets
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1,
                             width_shift_range=0.1, height_shift_range=0.1)
datagen.fit(X_train)

# Create a Sequential model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=15, validation_data=(X_test, y_test))

# Save the model
model.save('mnist_digit_recognition.h5')

def preprocess_image(image):
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the image
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    # Resize the image to 28x28 pixels
    image = cv2.resize(image, (28, 28))
    # Scale the pixel values to [0, 1] range
    image = image.astype("float32") / 255.0
    # Add a batch dimension
    image = np.expand_dims(image, axis=0)
    # Add a channel dimension
    image = np.expand_dims(image, axis=-1)
    return image

# Load the trained model
model = load_model('mnist_digit_recognition.h5')

# Load the image you want to classify
image = cv2.imread('digits.jpg')

# Apply some preprocessing to highlight the contours
blurred = cv2.GaussianBlur(image, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)

# Find contours in the image
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # Compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(contour)
    # Extract the digit and preprocess it
    digit = image[y:y + h, x:x + w]
    digit = preprocess_image(digit)
    # Make the prediction on the digit
    prediction = model.predict(digit).argmax(axis=1)
    # Draw the prediction on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, str(prediction[0]), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

# Save the image
cv2.imwrite('final_image.jpg', image)
