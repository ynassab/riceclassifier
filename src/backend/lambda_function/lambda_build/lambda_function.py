"""
Rice Classifier Backend

Serves as the backend API for the Rice Classifier web application. Processes
image data sent by the user, loads the trained model, performs image classification,
and returns the predicted rice variety along with confidence scores.

@author Yahia Nassab
"""

import tensorflow as tf
import numpy as np
import json
import os
import base64
from io import BytesIO
from PIL import Image

# Model configuration constants
NUM_PIXELS = 224
IMAGE_SIZE = (NUM_PIXELS, NUM_PIXELS)
KERNEL_SIZE = (4, 4)  # Larger-than-typical kernel for large features in rice
POOL_SIZE = (4, 4)  # Larger pool size to save more memory
LAMBDA_2 = 0.01
DROPOUT_RATE = 0.2
NUM_CLASS_NAMES = 5


def lambda_handler(event, context):
    """
    AWS Lambda entry point for rice grain classification API.

    This function serves as the main handler for HTTP requests to classify rice grain
    images using a pre-trained Convolutional Neural Network (CNN) model. It processes
    base64-encoded image data, loads the trained model, performs image classification,
    and returns the predicted rice variety along with confidence scores.

    Args:
        event (dict): AWS Lambda event object containing the HTTP request data.
                      Expected structure:
                      {
                         "body": {
                             "image": "<base64-encoded image data>"
                         }
                      }
                      The image should be in a standard format (PNG, JPEG, etc.)
                      and will be automatically resized to 224x224 pixels.

        context (LambdaContext): AWS Lambda context object containing runtime information.
                                 Not used in this implementation but required by Lambda.

    Returns:
        dict: HTTP response object with the following structure:
              Success (200):
              {
                  "statusCode": 200,
                  "body": "{
                      \"predicted_class\": <integer class index 0-4>,
                      \"all_predictions\": \"<comma-separated prediction scores>\",
                      \"re_encoded_image\": \"<base64-encoded processed image>\"
                  }"
              }

              Error (500):
              {
                  "statusCode": 500,
                  "body": "{\"error\": \"<error description>\"}"
              }

    Classification Classes:
        The model classifies rice grains into 5 varieties:
        - Class 0: Arborio
        - Class 1: Basmati
        - Class 2: Ipsala
        - Class 3: Jasmine
        - Class 4: Karacadag
    """
    try:
        model = create_model()
        checkpoint_dir = 'model'
        checkpoint_path = os.path.join(checkpoint_dir, 'ckpt_5')
        model.load_weights(checkpoint_path)

        if "body" in event:
            body = event['body']
            if isinstance(body, dict):
                body = body
            else:
                body = json.loads(body)
            image_data = body.get("image")  # Base64-encoded image

            if not image_data:
                raise ValueError("No image provided in the request.")

            decoded_image = Image.open(BytesIO(base64.b64decode(image_data)))
            input_image = preprocess_image(decoded_image)

            # Re-encode the decoded image for verification
            buffered = BytesIO()
            decoded_image.save(buffered, format="PNG")
            re_encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            predictions = model.predict(input_image)
            predicted_class = int(np.argmax(predictions[0]))  # Class index with highest probability

            response = {
                "statusCode": 200,
                "body": json.dumps({
                    "predicted_class": predicted_class,
                    "all_predictions": ', '.join([str(p) for p in predictions]),
                    "re_encoded_image": re_encoded_image
                    })
            }
            return response
        else:
            raise ValueError("No valid body in the request.")

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


def create_model():
    """
    Create and configure the Convolutional Neural Network model architecture for rice grain classification.

    This function builds a CNN model with an identical architecture to the one used during training,
    ensuring compatibility when loading pre-trained weights.

    Returns:
        tf.keras.Model: Compiled CNN model ready for training or inference.
                        Model expects input images of shape (224, 224, 3) and outputs
                        softmax probabilities for 5 rice grain classes.
    """
    data_augmentation = tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ])

    model = tf.keras.models.Sequential([
        data_augmentation,
        tf.keras.layers.Conv2D(32, kernel_size=KERNEL_SIZE, activation='relu', input_shape=IMAGE_SIZE + (3,)),
        tf.keras.layers.MaxPooling2D(pool_size=POOL_SIZE),
        tf.keras.layers.Conv2D(64, kernel_size=KERNEL_SIZE, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=POOL_SIZE),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(LAMBDA_2)),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(NUM_CLASS_NAMES, activation='softmax')
    ])
    return model


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess input image for CNN model inference.

    This function handles the complete preprocessing pipeline required to prepare
    user-uploaded images for the rice grain classification model. It ensures
    consistent input format regardless of the original image dimensions or colour channels.

    Args:
        image (PIL.Image): Input image object loaded from user data.
                           Can be in any standard format (JPEG, PNG, etc.)
                           and any dimensions.
        target_size (tuple, optional): Target dimensions for resizing as (width, height).
                                       Defaults to (224, 224) to match model input requirements.

    Returns:
        numpy.ndarray: Preprocessed image array ready for model inference.
                       Shape: (1, 224, 224, 3) - batch dimension added
                       Data type: float32, values typically in range [0, 255]
    """
    image = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)  # Convert to numpy array
    img_array = rgba2rgb(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def rgba2rgb(rgba, background=(255, 255, 255)):
    """
    Convert RGBA image array to RGB format by alpha blending with background colour.

    This utility function handles images with alpha channels (transparency) by compositing
    them over a specified background colour. This is necessary because the CNN model
    expects 3-channel RGB input, but user uploads may include PNG images with alpha channels.

    Args:
        rgba (numpy.ndarray): Input image array that may have 3 (RGB) or 4 (RGBA) channels.
                              Shape: (height, width, channels) where channels is 3 or 4.
                              Data type: typically uint8 with values 0-255.
        background (tuple, optional): RGB background colour for alpha blending.
                                      Defaults to (255, 255, 255) for white background.
                                      Each component should be in range [0, 255].

    Returns:
        numpy.ndarray: RGB image array with alpha channel removed (if present).
                       Shape: (height, width, 3)
                       Data type: uint8 with values 0-255.

    Alpha Blending Formula:
        For each RGB channel: result = foreground * alpha + background * (1 - alpha)
        Where alpha is normalized to range [0, 1].

    Adapted from the following Stack Overflow solution:
        https://stackoverflow.com/questions/50331463/convert-rgba-to-rgb-in-python
    """
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')
