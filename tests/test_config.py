import unittest
import shutil
from config import *

class TestConfig(unittest.TestCase):

    def test_path_constants_exist(self):
        """
        Test that all path-related constants are defined and are strings.
        """
        path_constants = [
            TRAINED_MODEL,
            TRAINING_PATH,
            SOURCE_PATH,
            HELMET_INPUT,
            HELMET_TRAINING,
            NO_HELMET_INPUT,
            NO_HELMET_TRAINING,
            TEST_IMAGE_PATH
        ]
        for constant in path_constants:
            self.assertIsInstance(constant, str)
            self.assertIsNotNone(constant)

    def test_derived_paths_correctly_formed(self):
        """
        Test that derived paths are correctly joined using os.path.join.
        """
        self.assertEqual(HELMET_INPUT, os.path.join(SOURCE_PATH, 'helmet'))
        self.assertEqual(HELMET_TRAINING, os.path.join(TRAINING_PATH, 'helmet'))
        self.assertEqual(NO_HELMET_INPUT, os.path.join(SOURCE_PATH, 'no_helmet'))
        self.assertEqual(NO_HELMET_TRAINING, os.path.join(TRAINING_PATH, 'no_helmet'))

    def test_model_configuration_values(self):
        """
        Test that model configuration constants have appropriate types and sensible ranges.
        """
        self.assertIsInstance(IMAGE_WIDTH, int)
        self.assertIsInstance(IMAGE_HEIGHT, int)
        self.assertIsInstance(IMAGE_COLORS, int)
        self.assertGreater(IMAGE_WIDTH, 0)
        self.assertGreater(IMAGE_HEIGHT, 0)
        self.assertIn(IMAGE_COLORS, [1, 3])

        self.assertIsInstance(LEARNING_RATE, float)
        self.assertGreater(LEARNING_RATE, 0)
        self.assertLess(LEARNING_RATE, 1)

        self.assertIsInstance(BATCH_SIZE, int)
        self.assertGreater(BATCH_SIZE, 0)
        self.assertLessEqual(BATCH_SIZE, 256)

        self.assertIsInstance(EPOCHS, int)
        self.assertGreater(EPOCHS, 0)

        self.assertIsInstance(VALIDATION_SPLIT, float)
        self.assertGreaterEqual(VALIDATION_SPLIT, 0)
        self.assertLess(VALIDATION_SPLIT, 1)

        self.assertIsInstance(METRIC, str)
        self.assertEqual(METRIC, 'accuracy')

        self.assertIsInstance(LOSS_FUNCTION, str)
        self.assertEqual(LOSS_FUNCTION, 'binary_crossentropy')

        self.assertIsInstance(THRESHOLD, float)
        self.assertGreaterEqual(THRESHOLD, 0)
        self.assertLessEqual(THRESHOLD, 1)

        self.assertIsInstance(REGULARIZER_STRENGTH, float)
        self.assertGreaterEqual(REGULARIZER_STRENGTH, 0)

        self.assertIsInstance(BRIGHTNESS_RANGE, list)
        self.assertEqual(len(BRIGHTNESS_RANGE), 2)
        self.assertIsInstance(BRIGHTNESS_RANGE[0], float)
        self.assertIsInstance(BRIGHTNESS_RANGE[1], float)
        self.assertLess(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
        self.assertGreater(BRIGHTNESS_RANGE[0], 0)

    def test_image_generation_configuration_values(self):
        """
        Test image generation constants.
        """
        self.assertIsInstance(CLASS_MODE, str)
        self.assertEqual(CLASS_MODE, 'binary')

    def test_augmentation_configuration_values(self):
        """
        Test augmentation configuration constants.
        """
        self.assertIsInstance(MULTIPLIER, int)
        self.assertGreaterEqual(MULTIPLIER, 1)

        self.assertIsInstance(ROTATION, (int, float))
        self.assertGreaterEqual(ROTATION, 0)

        self.assertIsInstance(ZOOM, (int, float, list))
        if isinstance(ZOOM, (int, float)):
            self.assertGreaterEqual(ZOOM, 0)
            self.assertLess(ZOOM, 1)
        elif isinstance(ZOOM, list):
            self.assertEqual(len(ZOOM), 2)
            self.assertGreaterEqual(ZOOM[0], 0)
            self.assertGreaterEqual(ZOOM[1], 0)
            self.assertLess(ZOOM[0], ZOOM[1])
            self.assertLessEqual(ZOOM[1], 1.0)

        self.assertIsInstance(SHEAR, (int, float))
        self.assertGreaterEqual(SHEAR, 0)

        self.assertIsInstance(HEIGHT_SHIFT, (int, float))
        self.assertGreaterEqual(HEIGHT_SHIFT, 0)
        self.assertLess(HEIGHT_SHIFT, 1)

        self.assertIsInstance(WIDTH_SHIFT, (int, float))
        self.assertGreaterEqual(WIDTH_SHIFT, 0)
        self.assertLess(WIDTH_SHIFT, 1)

        self.assertIsInstance(FILL_MODE, str)
        self.assertIn(FILL_MODE, ['constant', 'nearest', 'reflect', 'wrap'])

        self.assertIsInstance(HORIZONTAL_FLIP, bool)

        self.assertIsInstance(PHOTO_PREFIX, str)
        self.assertGreater(len(PHOTO_PREFIX), 0)

        self.assertIsInstance(ALLOWED_INPUT_FORMATS, list)
        self.assertGreater(len(ALLOWED_INPUT_FORMATS), 0)
        for fmt in ALLOWED_INPUT_FORMATS:
            self.assertIsInstance(fmt, str)
            self.assertGreater(len(fmt), 0)

        self.assertIsInstance(OUTPUT_FORMAT, str)
        self.assertGreater(len(OUTPUT_FORMAT), 0)


    def test_directory_creation_and_cleanup(self):
        """
        Test that the application can create the necessary directories based on config paths.
        This is a functional test that requires directory manipulation.
        """
        temp_training_path = "temp_" + TRAINING_PATH
        temp_source_path = "temp_" + SOURCE_PATH

        temp_helmet_training = os.path.join(temp_training_path, 'helmet')
        temp_no_helmet_training = os.path.join(temp_training_path, 'no_helmet')
        temp_helmet_input = os.path.join(temp_source_path, 'helmet')
        temp_no_helmet_input = os.path.join(temp_source_path, 'no_helmet')

        if os.path.exists(temp_training_path):
            shutil.rmtree(temp_training_path)
        if os.path.exists(temp_source_path):
            shutil.rmtree(temp_source_path)

        try:
            os.makedirs(temp_helmet_training)
            os.makedirs(temp_no_helmet_training)
            os.makedirs(temp_helmet_input)
            os.makedirs(temp_no_helmet_input)

            self.assertTrue(os.path.isdir(temp_helmet_training))
            self.assertTrue(os.path.isdir(temp_no_helmet_training))
            self.assertTrue(os.path.isdir(temp_helmet_input))
            self.assertTrue(os.path.isdir(temp_no_helmet_input))

        finally:
            if os.path.exists(temp_training_path):
                shutil.rmtree(temp_training_path)
            if os.path.exists(temp_source_path):
                shutil.rmtree(temp_source_path)

if __name__ == '__main__':
    unittest.main()