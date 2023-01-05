import os
import matplotlib.pylab as plt
import numpy as np

# Turn off info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


NUM_PIXELS        = 224
IMAGE_SIZE        = (NUM_PIXELS, NUM_PIXELS)
VALIDATION_SPLIT  = .20
SHUFFLING_SEED    = 123  # Arbitrary
BATCH_SIZE        = 16
NUM_EPOCHS        = 1


class Dataset():
	pass


def build_dataset(data_dir, subset):
	return tf.keras.preprocessing.image_dataset_from_directory(
		data_dir,
		validation_split=VALIDATION_SPLIT,
		subset=subset,
		label_mode="categorical",
		seed=SHUFFLING_SEED,  # Ensures validation set is stable between runs
		image_size=IMAGE_SIZE,
		batch_size=1)


def prepare_datasets_from_dir(data_dir):
	
	train_ds = Dataset()
	
	train_ds.data = build_dataset(data_dir, "training")
	train_ds.class_names = tuple(train_ds.data.class_names)
	train_ds.size = train_ds.data.cardinality().numpy()
	
	train_ds.data = train_ds.data.unbatch().batch(BATCH_SIZE)
	train_ds.data = train_ds.data.repeat()

	val_ds = Dataset()
	
	normalization_layer = tf.keras.layers.Rescaling(1. / 255)
	
	val_ds.data = build_dataset(data_dir, "validation")
	val_ds.size = val_ds.data.cardinality().numpy()
	
	val_ds.data = val_ds.data.unbatch().batch(BATCH_SIZE)
	val_ds.data = val_ds.data.map(lambda images, labels:
						(normalization_layer(images), labels))
	
	return train_ds, val_ds


def set_new_checkpoint_callback(checkpoint_dir):
	checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
	checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
				filepath=checkpoint_prefix,
				save_weights_only=True)
	return checkpoint_callback


def prepare_new_model(image_size, num_output_neurons):
	
	model = Sequential()
	
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_size + (3,)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())  # Unroll 3D output to 1D in order to pass to Dense layer
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_output_neurons, activation='softmax'))
	
	model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])
	
	return model

