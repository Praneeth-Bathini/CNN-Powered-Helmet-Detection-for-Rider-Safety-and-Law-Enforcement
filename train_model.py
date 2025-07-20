import os
import platform
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Sequential, models
from keras.src import regularizers
from keras.src.metrics import Precision, Recall, AUC
from keras.src.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import optimizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from utils import resize_images, ready_to_print, preprocessing, init_logger
from config import *
def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_COLORS),
                     padding='same', kernel_regularizer=regularizers.l1(REGULARIZER_STRENGTH)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l1(REGULARIZER_STRENGTH)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l1(REGULARIZER_STRENGTH)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l1(REGULARIZER_STRENGTH)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1(REGULARIZER_STRENGTH)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model
def compile_model(model):
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss=LOSS_FUNCTION,
                  metrics=["accuracy", Precision(), Recall(), AUC()])
def plot_metrics(history):
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
    loss, val_loss = history.history['loss'], history.history['val_loss']
    epochs = range(len(acc))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'g', label='Training Loss')
    plt.plot(epochs, val_loss, 'y', label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
def evaluate_model(model, validation_generator):
    val_labels = validation_generator.classes
    val_pred = model.predict(validation_generator, verbose=0)
    val_pred = (val_pred > 0.5).astype(int).reshape(-1)
    print("\nClassification Report:")
    print(classification_report(val_labels, val_pred, target_names=list(validation_generator.class_indices.keys())))
    cm = confusion_matrix(val_labels, val_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=validation_generator.class_indices,
                yticklabels=validation_generator.class_indices)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
def train_model():
    logger = init_logger()
    model = get_model()
    compile_model(model)
    datagen = ImageDataGenerator(rescale=1./255, validation_split=VALIDATION_SPLIT)
    train_generator = datagen.flow_from_directory(
        TRAINING_PATH, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE, shuffle=True, subset='training')
    val_generator = datagen.flow_from_directory(
        TRAINING_PATH, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE, shuffle=False, subset='validation')
    class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
    class_weights = dict(enumerate(class_weights))
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
        ModelCheckpoint(filepath=TRAINED_MODEL, monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=1e-7),
        TensorBoard(log_dir="logs")
    ]
    history = model.fit(train_generator,
                        steps_per_epoch=train_generator.samples // BATCH_SIZE,
                        validation_data=val_generator,
                        validation_steps=val_generator.samples // BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=callbacks,
                        class_weight=class_weights)
    model.save(TRAINED_MODEL)
    plot_metrics(history)
    evaluate_model(model, val_generator)
if __name__ == "__main__":
    train_model()