import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image


def init_logger(log_file: str = "training.log"):
    """
    Initialize and configure a logger with both console and file handlers.

    Creates a logger that outputs to both stdout and a specified log file with the following format:
    TIMESTAMP - LOGLEVEL - MODULE_NAME - FUNCTION_NAME - MESSAGE

    :param log_file: Path to the log file. Default "training.log" in the current directory
    :return: Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger("root_logger")


def ready_to_print(timestamp) -> bool:
    """
    Returns True if the current time is greater than the given timestamp.

    :param timestamp: Given timestamp from datetime.
    :return: True if the current time is greater than the given timestamp otherwise False.
    """
    return datetime.now().timestamp() >= timestamp


def resize_images(input_folder: str, output_folder: str, target_size: (int, int), logger: logging.Logger):
    """
    Resize images and paste them onto a new image with a white background while keeping their aspect ratio.

    :param input_folder: Path to the folder with images.
    :param output_folder: Path to the folder where resized images will be saved.
    :param target_size: (width, height).
    :param logger: Reference to a root logger object.
    :return: None
    """
    os.makedirs(output_folder, exist_ok=True)

    timestamp = datetime.now().timestamp()
    file_number: int = 0
    total_pictures: int = len(os.listdir(input_folder))
    for filename in os.listdir(input_folder):
        file_number += 1
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(input_folder, filename)
            preprocessing(image_path, target_size[0], target_size[1], logger, output_folder,
                          is_logging=False, output_file_name_prefix="")
            if ready_to_print(timestamp):
                logger.info(f'Resized pictures: {file_number}/{total_pictures}')
                timestamp = (datetime.now() + timedelta(seconds=1)).timestamp()
        else:
            logger.info(f'Skipping non supported file: {filename} - {file_number}/{total_pictures}')
    logger.info(f'All pictures resized: {file_number}/{total_pictures}')

def preprocessing(image_path: str, width: int, height: int, logger: logging.Logger,
                  output_folder: str, is_logging: bool = True, output_file_name_prefix: str = "preprocessed_"):
    """
    Preprocesses an image by resizing it to fit within the specified dimensions
    while maintaining its aspect ratio and padding it with a white background
    to match the exact target dimensions.

    The processed image is saved in the same directory as the original image,
    with a filename prefixed by 'preprocessed_'.


    :param image_path: The absolute or relative path to the image file.
    :param width: The target width for the preprocessed image in pixels.
    :param height: The target height for the preprocessed image in pixels.
    :param logger: Reference to a root logger object.
    :return: The path to the saved preprocessed image.
    :param output_folder: Path to the folder where resized images will be saved.
    :param output_file_name_prefix: Prefixed for newly generated filename.
    :param is_logging: Individual log for each preprocessed image.
    """
    selected_image = Image.open(image_path)
    selected_image.thumbnail((width, height),
                             Image.Resampling.LANCZOS)  # Resize the image while maintaining the aspect ratio
    new_image = Image.new("RGB", (width, height), (255, 255, 255))
    position = ((width - selected_image.size[0]) // 2, (height - selected_image.size[1]) // 2)
    new_image.paste(selected_image, position)
    output_path = os.path.join(output_folder if output_folder else Path(image_path).parent,
                               output_file_name_prefix + os.path.basename(image_path))
    new_image.save(output_path)

    if is_logging:
        logger.info(f"Resized and saved {output_path}")
    return output_path
