#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from glob import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# In[ ]:


# Parameters
num_classes = 10  # Adjust based on your number of classes
image_size = (224, 224)  # EfficientNet default input size for B0
batch_size = 32
epochs = 10

# Paths
data_dir = 'path_to_your_data'  # Update with your dataset path

# Load dataset
def load_dataset(data_dir, image_size):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    class_indices = {class_name: i for i, class_name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for img_path in glob(os.path.join(class_dir, '*.jpg')):  # Assuming images are in .jpg format
            img = load_img(img_path, target_size=image_size)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(class_indices[class_name])
    
    images = np.array(images, dtype='float32')
    labels = np.array(labels)
    return images, labels, class_names

images, labels, class_names = load_dataset(data_dir, image_size)


# In[ ]:


# Normalize images
images /= 255.0

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

# Create ImageDataGenerators
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
test_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)

# Load EfficientNetB0 with pre-trained ImageNet weights, excluding the top layers
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=image_size + (3,))

# Freeze the base model
base_model.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Added dropout layer for regularization
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)


# In[ ]:


# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Model summary
model.summary()

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=test_generator,
    validation_steps=len(test_generator),
    epochs=epochs
)


# In[ ]:


# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Get the true labels and predictions
Y_pred = model.predict(test_generator, steps=len(test_generator))
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print('Classification Report')
print(classification_report(y_true, y_pred, target_names=class_names))

