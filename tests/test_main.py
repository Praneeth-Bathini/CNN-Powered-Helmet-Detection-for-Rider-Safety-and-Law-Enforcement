import unittest
from unittest.mock import patch
import numpy as np
from tensorflow.keras import models

from main import get_adam, get_model
from utils import ready_to_print

class TestFunctions(unittest.TestCase):

    def test_get_model(self):
        model = get_model()
        self.assertIsInstance(model, models.Sequential)

    @patch("platform.processor", return_value="x86")
    @patch("platform.system", return_value="Linux")
    def test_get_adam(self, mock_processor, mock_system):
        optimizer = get_adam()
        self.assertEqual(optimizer.__class__.__name__, "Adam")

    def test_ready_to_print(self):
        timestamp = np.datetime64('2025-05-01T00:09').astype('datetime64[s]').astype(float)
        self.assertTrue(ready_to_print(timestamp))

if __name__ == "__main__":
    unittest.main()
