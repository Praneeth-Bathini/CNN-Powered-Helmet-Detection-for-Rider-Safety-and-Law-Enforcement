import unittest
import os
import logging
import shutil
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from PIL import Image
import sys

from utils import init_logger, ready_to_print, resize_images, preprocessing


def _create_dummy_image(path, size=(100, 100), color=(0, 0, 0)):
    """Helper to create a dummy image file."""
    if path.endswith((".png", ".jpg")):
        img = Image.new('RGB', size, color)
        img.save(path)
    else:
        with open(path, 'w') as f:
            f.write("dummy content")


class TestUtils(unittest.TestCase):

    def setUp(self):
        """Set up for tests, creating temporary directories and a logger."""
        self.test_log_file = "test_training.log"
        self.logger = logging.getLogger("root_logger")
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
        self.logger.setLevel(logging.INFO)

        self.test_input_dir = "test_input_images"
        self.test_output_dir = "test_output_images"
        os.makedirs(self.test_input_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)

        _create_dummy_image(os.path.join(self.test_input_dir, "test_image1.png"), (100, 150))
        _create_dummy_image(os.path.join(self.test_input_dir, "test_image2.jpg"), (200, 100))
        _create_dummy_image(os.path.join(self.test_input_dir, "test_image3.txt"), (50, 50))  # Non-image file

    def tearDown(self):
        """Clean up after tests, removing temporary directories and log file."""
        if os.path.exists(self.test_input_dir):
            shutil.rmtree(self.test_input_dir)
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)

    @patch('logging.basicConfig')
    @patch('logging.getLogger')
    def test_init_logger(self, mock_get_logger, mock_basic_config):
        """Test logger initialization and configuration."""
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance

        logger = init_logger("custom_log.log")

        args, kwargs = mock_basic_config.call_args
        self.assertEqual(kwargs['level'], logging.INFO)
        self.assertEqual(kwargs['format'], '%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

        handlers = kwargs['handlers']
        self.assertEqual(len(handlers), 2)

        self.assertIsInstance(handlers[0], logging.StreamHandler)
        self.assertEqual(handlers[0].stream, sys.stdout)

        self.assertIsInstance(handlers[1], logging.FileHandler)
        self.assertEqual(os.path.abspath(handlers[1].baseFilename), os.path.abspath("custom_log.log"))

        mock_get_logger.assert_called_once_with("root_logger")
        self.assertEqual(logger, mock_logger_instance)


        mock_basic_config.reset_mock()
        mock_get_logger.reset_mock()
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance

        init_logger()

        args, kwargs = mock_basic_config.call_args
        self.assertEqual(kwargs['level'], logging.INFO)
        self.assertEqual(kwargs['format'], '%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

        handlers = kwargs['handlers']
        self.assertEqual(len(handlers), 2)
        self.assertIsInstance(handlers[0], logging.StreamHandler)
        self.assertEqual(handlers[0].stream, sys.stdout)
        self.assertIsInstance(handlers[1], logging.FileHandler)
        self.assertEqual(os.path.abspath(handlers[1].baseFilename), os.path.abspath("training.log"))

    def test_ready_to_print(self):
        """Test ready_to_print function's time comparison."""
        now = datetime.now()
        self.assertTrue(ready_to_print(now.timestamp()))

        past_timestamp = (now - timedelta(seconds=5)).timestamp()
        self.assertTrue(ready_to_print(past_timestamp))

        future_timestamp = (now + timedelta(seconds=5)).timestamp()
        self.assertFalse(ready_to_print(future_timestamp))

    @patch('utils.preprocessing')
    @patch('utils.ready_to_print', return_value=True)
    def test_resize_images(self, _, mock_preprocessing):
        """Test resize_images functionality."""
        target_size = (128, 128)
        input_folder = self.test_input_dir
        output_folder = self.test_output_dir

        with self.assertLogs(self.logger.name, level='INFO') as cm:
            resize_images(input_folder, output_folder, target_size, self.logger)

            self.assertEqual(mock_preprocessing.call_count, 2)  # Two valid image files created in setUp
            calls = mock_preprocessing.call_args_list
            self.assertIn(os.path.join(input_folder, "test_image1.png"), [call.args[0] for call in calls])
            self.assertIn(os.path.join(input_folder, "test_image2.jpg"), [call.args[0] for call in calls])

            self.assertIn('All pictures resized: 3/3', cm.output[-1])
            self.assertIn('Skipping non supported file:', cm.output[-4])

            self.assertTrue(os.path.exists(output_folder))
            self.assertTrue(os.path.isdir(output_folder))

    def test_preprocessing(self):
        """Test preprocessing function for image resizing and padding."""
        input_image_path = os.path.join(self.test_input_dir, "test_image_pre.png")
        _create_dummy_image(input_image_path, (50, 100))
        target_width = 128
        target_height = 128

        with self.assertLogs(self.logger.name, level='INFO') as cm:
            output_path = preprocessing(input_image_path, target_width, target_height, self.logger,
                                        self.test_output_dir, is_logging=True)

            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(os.path.isfile(output_path))

            with Image.open(output_path) as img:
                self.assertEqual(img.size, (target_width, target_height))
                self.assertEqual(img.getpixel((0, 0)), (255, 255, 255))
                self.assertEqual(img.getpixel((39, 14)), (0, 0, 0))

            self.assertIn(f"Resized and saved {output_path}", cm.output[0])

        mock_logger_info = MagicMock()
        with patch.object(self.logger, 'info', mock_logger_info):
            preprocessing(input_image_path, target_width, target_height, self.logger, self.test_output_dir,
                          is_logging=False)
            mock_logger_info.assert_not_called()

        output_path_custom_prefix = preprocessing(
            input_image_path, target_width, target_height, self.logger, self.test_output_dir,
            is_logging=False, output_file_name_prefix="custom_"
        )
        self.assertTrue("custom_test_image_pre.png" in output_path_custom_prefix)
        self.assertTrue(os.path.exists(output_path_custom_prefix))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)