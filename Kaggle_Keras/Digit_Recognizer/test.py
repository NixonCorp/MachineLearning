from keras.models import load_model
from keras import backend as K
from keras.utils import np_utils
import numpy as np
import pandas as pd


VERBOSE = 1
NB_CLASSES = 10

# read train and test data, cpnvert to 3d array
X_train_with_label = pd.read_csv('input/train.csv')
labels = X_train_with_label.values[:, 0].astype('int32')
X_train = X_train_with_label.drop('label', axis=1)
X_test = pd.read_csv('input/test.csv')
X_train = X_train.values.reshape(42000, 28, 28)
X_test = X_test.values.reshape(28000, 28, 28)
K.set_image_dim_ordering("th")

# consider them as float and normalize
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# we need a 60K x [1 x 28 x 28] shape as input to the CNN
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]

y_train = np_utils.to_categorical(labels, NB_CLASSES)

model = load_model("models/model-17.h5")
score = model.evaluate(X_train, y_train, verbose=VERBOSE)
print("\nTrain loss:", score[0])
print('Train accuracy:', score[1])
predictions = model.predict_classes(X_test, verbose=VERBOSE)

result = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)),
                       "Label": predictions})

result.to_csv("./pred_mnist.csv",
              columns=('ImageId', 'Label'),
              index=False)
