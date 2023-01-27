# rice-ml
The rice image dataset is provided courtesy of Kaggle user MURAT KOKLU under a CC0 Public Domain license. The dataset may be downloaded and used as-is for this program here:

https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

https://www.muratkoklu.com/datasets/

The best checkpoint for this model, to be loaded from the directory `training_checkpoints_1/`, can be downloaded at https://yahianassab.com/download-rice-ml.html

## Model overview
The model is a typical Convolutional Neural Network (CNN) scheme: it begins with two pairs of stacked Conv2D and MaxPool2D layers, the output of which is flattened and passed to two Dense layers. The five output neurons correspond to the five classes of rice to be classified.

## Limitations
The current iteration of the model fits its training data with an accuracy of 96%, but is notably worse (although better than chance, ~34%) when making inferences on new data, indicating that it is overfit. Further training can rapidly and quite repeatably increase the training accuracy to ~99.7%, but this exacerbates the overfitting problem.

## Possible Solutions
The most probable solution is to decrease the model's capacity (i.e., to use a simpler model with fewer trainable parameters). This will require significant experimentation, but a good place to start would be a single stack of Conv2D-MaxPool2D layers, followed by a much smaller Dense layer than the one currently used (16 or 32 neurons instead of 128, for example), followed by an output layer.

Another solution would be to use a pre-trained model from Tensorflow-Hub and extend it. This is non-trivial, however, since a model such as the `efficientnetv2-b0-21k-ft1k` model, which works for images with 224-pixel width, will quickly converge to a training accuracy of ~47% while retaining a validation accuracy of chance (~20%).

Another solution would be to increase the size of the training data, since 75000 images is a relatively small sample. However, this is not possible for this application, so it is a limitation that must be worked around.
