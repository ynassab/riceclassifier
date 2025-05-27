import tensorflow as tf
import numpy as np
import json
import os
import base64
from io import BytesIO
from PIL import Image

NUM_PIXELS = 224
IMAGE_SIZE = (NUM_PIXELS, NUM_PIXELS)
KERNEL_SIZE = (4, 4)  # Larger-than-typical kernel for large features in rice
POOL_SIZE = (4, 4)  # Larger pool size to save more memory
LAMBDA_2 = 0.01
DROPOUT_RATE = 0.2
NUM_CLASS_NAMES = 5


def lambda_handler(event, context):
    try:
        model = create_model()
        checkpoint_dir = 'model'
        checkpoint_path = os.path.join(checkpoint_dir, 'ckpt_5')
        model.load_weights(checkpoint_path)

        if "body" in event:
            body = event['body']
            if isinstance(body, dict):
                body = body  # Already a dictionary, no need to parse
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
    image = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)  # Convert to numpy array
    img_array = rgba2rgb(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def rgba2rgb( rgba, background=(255,255,255) ):
    # https://stackoverflow.com/questions/50331463/convert-rgba-to-rgb-in-python

    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )
