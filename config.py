import os

# Paths
TRAINED_MODEL = 'trained_model.h5'
TRAINING_PATH = 'training'
SOURCE_PATH = 'source'
HELMET_INPUT = os.path.join(SOURCE_PATH, 'helmet')
HELMET_TRAINING = os.path.join(TRAINING_PATH, 'helmet')
NO_HELMET_INPUT = os.path.join(SOURCE_PATH, 'no_helmet')
NO_HELMET_TRAINING = os.path.join(TRAINING_PATH, 'no_helmet')
TEST_IMAGE_PATH = "demo_images/helmet_0.jpg"


# Model configuration
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_COLORS = 3
LEARNING_RATE = 0.00005
BATCH_SIZE = 64
EPOCHS = 100
VALIDATION_SPLIT = 0.2  # 20% of the data will be used for validation
METRIC = 'accuracy'
LOSS_FUNCTION = 'binary_crossentropy'
THRESHOLD = 0.5
REGULARIZER_STRENGTH = 0.0

# Image generation
CLASS_MODE = 'binary'

# Augmentation configuration
MULTIPLIER = 10
ROTATION = 35
ZOOM = 0.1
SHEAR = 0.2
HEIGHT_SHIFT = 0
WIDTH_SHIFT = 0.2
FILL_MODE = 'reflect'
HORIZONTAL_FLIP = True
PHOTO_PREFIX = 'generated'
ALLOWED_INPUT_FORMATS = ['png', 'jpg']
OUTPUT_FORMAT = 'jpg'
BRIGHTNESS_RANGE = [0.7, 1.3]