import unittest
import base64
import json
from io import BytesIO
from PIL import Image
import numpy as np

from backend.lambda_function.lambda_build import lambda_function


def make_sample_image_base64():
    """Helper: create a simple 224x224 RGB image and return as base64 string."""
    img = Image.new("RGB", (224, 224), color=(255, 0, 0))  # Red square
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class MockModel:
    """Fake TensorFlow model for testing."""
    def load_weights(self, path):
        return None

    def predict(self, x):
        # Return fixed probabilities (batch size 1, 5 classes)
        return np.array([[0.1, 0.2, 0.3, 0.15, 0.25]])


class TestLambdaHandler(unittest.TestCase):
    def setUp(self):
        """Patch create_model before each test."""
        patcher = unittest.mock.patch.object(
            lambda_function,
            "create_model",
            return_value=MockModel()
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_lambda_handler_success(self):
        """Test successful classification flow."""
        event = {
            "body": json.dumps({
                "image": make_sample_image_base64()
            })
        }
        response = lambda_function.lambda_handler(event, None)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])

        # Check keys exist
        assert "predicted_class" in body
        assert "all_predictions" in body
        assert "re_encoded_image" in body

        # Prediction should match mocked np.argmax -> class 2
        assert body["predicted_class"] == 2

    def test_lambda_handler_no_body(self):
        """Test when no body is provided in event."""
        event = {}
        response = lambda_function.lambda_handler(event, None)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "error" in body


    def test_lambda_handler_no_image(self):
        """Test when body exists but no image provided."""
        event = {
            "body": json.dumps({})
        }
        response = lambda_function.lambda_handler(event, None)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "error" in body
        assert "No image" in body["error"]
