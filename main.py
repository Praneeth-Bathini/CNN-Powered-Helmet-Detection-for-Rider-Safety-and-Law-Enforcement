import platform
from datetime import datetime, timedelta
from pathlib import Path

import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential, models
from keras.src import regularizers
from keras.src.metrics import Precision, Recall, AUC
from keras.src.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import optimizers

from sklearn.utils.class_weight import compute_class_weight

from utils import resize_images, ready_to_print, preprocessing, init_logger

from config import *


def augment_pictures(multiplier: int, input_path: str, train_dir_path: str):
    """
    This function generates augmented pictures from the original ones.

    :param multiplier: Number of augmented pictures to generate for each original picture.
    :param input_path: Path to the original pictures.
    :param train_dir_path: Path to the directory where the augmented pictures will be saved.
    :return: None
    """
    # Change all pictures to a uniform size (IMAGE_WIDTH, IMAGE_HEIGHT)
    resized_folder_path = os.path.join('resized', os.path.basename(input_path))
    resize_images(input_path, resized_folder_path, (IMAGE_WIDTH, IMAGE_HEIGHT), logger)

    # Augmentation configuration
    generator = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=ROTATION,
                                   width_shift_range=WIDTH_SHIFT,
                                   height_shift_range=HEIGHT_SHIFT,
                                   shear_range=SHEAR,
                                   zoom_range=ZOOM,
                                   horizontal_flip=HORIZONTAL_FLIP,
                                   fill_mode=FILL_MODE,
                                   brightness_range=BRIGHTNESS_RANGE,
                                   )
    # Load only allowed pictures paths
    Path(train_dir_path).mkdir(parents=True, exist_ok=True)
    generated_pictures: int = 0
    pictures_path: list[str] = []
    with os.scandir(resized_folder_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.split(".")[-1] in ALLOWED_INPUT_FORMATS:
                pictures_path.append(entry.path)
    timestamp = datetime.now().timestamp()
    # Augment them one by one
    for path in pictures_path:
        if ready_to_print(timestamp):
            logger.info(f'Generated pictures: {generated_pictures}/{len(pictures_path)}')
            timestamp = (datetime.now() + timedelta(seconds=1)).timestamp()
        generated_pictures += 1
        pic = load_img(path)
        pic_array = img_to_array(pic)
        pic_array = pic_array.reshape((1,) + pic_array.shape)
        count = 0
        for _ in generator.flow(pic_array, batch_size=BATCH_SIZE,
                                save_to_dir=train_dir_path,
                                save_prefix=PHOTO_PREFIX, save_format=OUTPUT_FORMAT):
            count += 1
            if count == multiplier:
                break
    logger.info(f'All pictures generated: {generated_pictures}/{len(pictures_path)}')


def get_model():
    """
    This function constructs a convolutional neural network (CNN) using Keras' Sequential API for binary classification
    tasks. The architecture follows a classic CNN design pattern with convolutional blocks followed by dense

    :return: Sequential Keras model.
    """
    model = Sequential()

    # Block 1
    model.add(
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_COLORS), padding='same', kernel_regularizer=regularizers.l1(REGULARIZER_STRENGTH)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.1))
    model.add(MaxPooling2D((2, 2)))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(REGULARIZER_STRENGTH)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.1))
    model.add(MaxPooling2D((2, 2)))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(REGULARIZER_STRENGTH)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.1))
    model.add(MaxPooling2D((2, 2)))

    # Block 4
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(REGULARIZER_STRENGTH)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    # Flatten and Dense layers
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1(REGULARIZER_STRENGTH)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.summary()
    return model


def get_adam():
    """
    Gets the appropriate Adam optimizer instance depending on the system and processor.

    On macOS systems with ARM processors, this function attempts to use the legacy
    Adam optimizer, if available. If the legacy optimizer is unavailable, it falls
    back to the standard TensorFlow Adam optimizer. For other platforms, the standard
    TensorFlow Adam optimizer is returned by default.

    :return: An instance of TensorFlow's Adam optimizer, which could either be the
        legacy version or the standard version, depending on the platform and
        processor.
    """
    if platform.processor() == 'arm' and platform.system() == 'Darwin':
        try:
            return optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
        except AttributeError:
            return optimizers.Adam(learning_rate=LEARNING_RATE)
    else:
        return optimizers.Adam(learning_rate=LEARNING_RATE)


def compile_model(raw_model):
    """
    Compiles a given Keras model with predefined training parameters (optimizer - Adam, loss function - binary_crossentropy, and metrics - accuracy).

    :param raw_model: Uncompiled Keras model to be compiled.
    :return: Compiled a Keras model ready for training
    """
    raw_model.compile(
        optimizer=get_adam(),
        loss=LOSS_FUNCTION,
        metrics=[METRIC, Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
    )


def fit_model(model, train_generator, validation_generator):
    """
    Trains a given model using training and validation data generators.

    :param model: A compiled Keras model ready for training.
    :param train_generator: ImageDataGenerator for training data.
    :param validation_generator: ImageDataGenerator for validation data.
    :return: Object containing training/validation metrics history.
    """
    train_labels = train_generator.classes  # Get labels from the generator
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = dict(enumerate(class_weights))
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=TRAINED_MODEL, monitor='val_accuracy', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=1e-7)
    result = model.fit(train_generator,
                       validation_split=VALIDATION_SPLIT,
                       steps_per_epoch=train_generator.samples // BATCH_SIZE,
                       validation_data=validation_generator,
                       validation_steps=validation_generator.samples // BATCH_SIZE,
                       epochs=EPOCHS,
                       callbacks=[early_stopping, model_checkpoint, lr_scheduler],
                       class_weight=class_weights,
                       verbose=1)
    return result


def plot_graph(history):
    """
    Plots training and validation accuracy/loss graphs from a Keras history object.

    :param history: A Keras History object containing training metrics history
    :return: None (displays matplotlib plot)
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(10, 5))  # Adjust the figure size if needed

    plt.subplot(1, 2, 1)  # Create a subplot for the accuracy
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)  # Create a subplot for the loss
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'y', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()  # Adjust spacing between subplots

    plt.show()


def demo(test_image_path: str):
    """
    Demonstrates helmet detection on a test image.

    Loads a trained model, preprocesses the test image, makes a prediction, and displays the result along with the image.

    :return: None
    """
    try:
        model = models.load_model(TRAINED_MODEL)
    except OSError as e:
        raise FileNotFoundError(f"Model file not found at {TRAINED_MODEL}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading model from {TRAINED_MODEL}: {str(e)}") from e
    if not os.path.exists(test_image_path):
        raise RuntimeError(f"Preprocessed image not found at {test_image_path}")
    preprocessed_path = preprocessing(test_image_path, IMAGE_WIDTH, IMAGE_HEIGHT, logger, "demo_images")
    img = image.load_img(preprocessed_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0
    images = np.vstack([x])
    classes = model.predict(images, batch_size=1)
    print("Prediction: " + str(classes[0][0]) + " (0 = helmet, 1 = no helmet)")
    if classes[0][0] > THRESHOLD:
        print("There is likely driver without helmet!")
    else:
        print("There is likely driver with helmet!")
    plt.imshow(img)


def train_and_save_model():
    """
    Trains and saves a machine learning model using image data generators.

    This function:
    1. Retrieves a pre-configured model architecture
    2. Compiles the model with appropriate loss and metrics
    3. Sets up data generators for training and validation with augmentation
    4. Trains the model on the generated data
    5. Saves the trained model to disk
    6. Plots training history graphs

    :return: None
    """
    model = get_model()
    compile_model(model)
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=VALIDATION_SPLIT)
    train_generator = datagen.flow_from_directory(
        directory=TRAINING_PATH,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        shuffle=True,
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        directory=TRAINING_PATH,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        shuffle=False,
        subset='validation'
    )

    history = fit_model(model, train_generator, validation_generator)
    model.save(TRAINED_MODEL, include_optimizer=True)
    plot_graph(history)


def parse_args():
    """
    Parse command line arguments for the helmet detection software.

    :return: Namespace object containing the parsed arguments
    """
    parser = argparse.ArgumentParser(description='Helmet detection software')
    parser.add_argument('image_path', nargs='?', default=TEST_IMAGE_PATH,
                        help='Path to the image file to be analysed (default: %(default)s)')
    return parser.parse_args()


if __name__ == "__main__":
    logger = init_logger()
    logger.info("Starting...")
    if not os.path.exists(NO_HELMET_TRAINING):
        logger.info("Augmentation of driver without helmet")
        augment_pictures(MULTIPLIER, NO_HELMET_INPUT, NO_HELMET_TRAINING)
    if not os.path.exists(HELMET_TRAINING):
        logger.info("Augmentation of driver with helmet")
        augment_pictures(MULTIPLIER, HELMET_INPUT, HELMET_TRAINING)
    if not os.path.exists(TRAINED_MODEL):
        logger.info("Training model")
        train_and_save_model()
    demo(parse_args().image_path)
