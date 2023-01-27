# rice-ml
The rice image dataset is provided courtesy of Kaggle user MURAT KOKLU under a CC0 Public Domain license. https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

In the featured notebook, the dataset is downloaded directly from the dataset contributor's website. Details of this and other datasets can be found here: https://www.muratkoklu.com/datasets/

## Model overview
The model architecture is that of a typical low-capacity Convolutional Neural Network (CNN): it begins with two pairs of stacked image convolution layers and max-pooling layers, the output of which is flattened and passed to two fully connected layers (including the output layer).

## Very high validation accuracy
The model predicts classifications on data it has not seen before with an accuracy of 98.0%. To avoid overfitting ("memorizing" the training data), the following features were implemented in the model:

1. Data augmentation: each training image is randomly flipped and rotated before being passed to the model as input.
2. L2 regularization: the hidden fully connected layer incurs a loss penalty when updating its weights. This promotes weight updates that are transferable between images, instead of being applicable to only one image.
3. Dropout: 20% of the weights of the hidden layer are "dropped" (set to zero) before being passed to the output layer, promoting weights that are recurring and persistent between training steps.

## Potential future implementations
It may be possible to achieve an even higher validation accuracy through transfer learning (i.e., building upon the foundation of a pre-trained model). However, the performance of the current model is already satisfactory, largely as a result of the consistency of the training dataset (monochrome objects of interest centred and set against a solid black background with minimal variability). Transfer learning techniques, meanwhile, excel where the target data is much less regular. Nonetheless, an example base model that could be used would be `efficientnetv2-b0-21k-ft1k` from _tensorflow-hub_, which accepts images of 224-pixel width, as with the images in this dataset.
