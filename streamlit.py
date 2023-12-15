import streamlit as st
from matplotlib import pyplot as plt
import os
import random
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory

# global variables:
batch_size = 32
image_size = (64, 64)
validation_split = 0.2

# Create the training dataset from the 'train' directory
train_ds = image_dataset_from_directory(
    directory='datasets/training_set',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='training',
    seed=123
)

# Create the validation dataset from the 'train' directory
validation_ds = image_dataset_from_directory(
    directory='datasets/training_set',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='validation',
    seed=123
)

# Create the testing dataset from the 'test' directory
test_ds = image_dataset_from_directory(
    directory='datasets/test_set',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size
)

NUM_CLASSES = 5
IMG_SIZE = 64
# There is no shearing option anymore, but there is a translation option
HEIGTH_FACTOR = 0.2
WIDTH_FACTOR = 0.2

# Create a sequential model with a list of layers
model = tf.keras.Sequential([
  # Add a resizing layer to resize the images to a consistent shape
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  # Add a rescaling layer to rescale the pixel values to the [0, 1] range
  layers.Rescaling(1./255),
  # Add some data augmentation layers to apply random transformations during training
  layers.RandomFlip("horizontal"),
  layers.RandomTranslation(HEIGTH_FACTOR,WIDTH_FACTOR),
  layers.RandomZoom(0.2),

  layers.Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation="relu"),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Conv2D(32, (3, 3), activation="relu"),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation="relu"),
  layers.Dense(NUM_CLASSES, activation="softmax")
])

# Compile and train your model as usual
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# EDA functions
def count_images_in_folders(safe_folder):
    parent_dir = "C:/Users/Jarne/Documents/schooljaar 2023-2024/ai/task3/datasets"
    path = os.path.join(parent_dir, safe_folder)

    if not os.path.exists(path) or not os.path.isdir(path):
        print(f"Invalid directory: {path}")
        return
     
    classes = [class_name for class_name in os.listdir(path) if os.path.isfile(os.path.join(path, class_name))]
    
    print(f"The folder: {safe_folder} has: {len(classes)} images")

def showRandom2Images(safe_folder, photo_name):
    parent_dir = "C:/Users/Jarne/Documents/schooljaar 2023-2024/ai/task3/datasets"
    path = os.path.join(parent_dir, safe_folder)

    images = []

    for i in range(2):
        rnd = random.randint(0,len([class_name for class_name in os.listdir(path) if os.path.isfile(os.path.join(path, class_name))])-1)
        img_orig = cv2.imread(path + '/' + photo_name + str(rnd) + '.jpg')
        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)

    plt.figure(figsize = (5, 10))
    for i in range(2):
        plt.subplot(1,2 ,i+1)
        plt.imshow(images[i])
        plt.axis('off')

# Model training function
def train_model(train_ds, validation_ds, epochs=100):
    steps_per_epoch = len(train_ds)
    global history
    history = model.fit(train_ds,
                validation_data = validation_ds,
                steps_per_epoch = steps_per_epoch,
                epochs = epochs,
                verbose=1,  # Set to 1 to see training progress
                callbacks=[
                  # Add early stopping to prevent overfitting
                  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),

                  # Add learning rate scheduling
                  tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7),

                  # Add model checkpoint to save the best weights during training
                  tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
              ]
                )
    

# Visualize training function
def visualize_training(history):
    # Create a figure and a grid of subplots with a single call
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    # Plot the loss curves on the first subplot
    ax1.plot(history.history['loss'], label='training loss')
    ax1.plot(history.history['val_loss'], label='validation loss')
    ax1.set_title('Loss curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot the accuracy curves on the second subplot
    ax2.plot(history.history['accuracy'], label='training accuracy')
    ax2.plot(history.history['val_accuracy'], label='validation accuracy')
    ax2.set_title('Accuracy curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Show the figure
    plt.show()

def main():
    st.title("Your Streamlit App")

    # EDA Section
    st.header("Exploratory Data Analysis")
    st.subheader("Image Counts in Folders")
    count_images_in_folders(football_folder_name)
    count_images_in_folders(basketball_folder_name)
    count_images_in_folders(tennis_folder_name)
    count_images_in_folders(golf_folder_name)
    count_images_in_folders(volleyball_folder_name)

    st.subheader("Random Images")
    showRandom2Images(football_folder_name, football_photo_name)
    showRandom2Images(basketball_folder_name, basketball_photo_name)
    showRandom2Images(tennis_folder_name, tennisball_photo_name)
    showRandom2Images(golf_folder_name, golfball_photo_name)
    showRandom2Images(volleyball_folder_name, volleyball_photo_name)

    # Model Training Section
    st.header("Model Training")
    # Add controls for training parameters (e.g., epochs)
    epochs = st.slider("Number of Epochs", min_value=1, max_value=100, value=20)
    st.info(f"Training for {epochs} epochs...")
    train_model(train_ds, validation_ds, epochs)

    # Visualize Training Section
    st.header("Visualize Training")
    # Visualize training curves
    visualize_training(history)

if __name__ == '__main__':
    main()
