
import os
import sys

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import tensorflow        as tf
import tensorflow_hub    as hub

import ssl

from keras                                   import backend as K
from keras.models                            import Model
from keras.preprocessing                     import image_dataset_from_directory
from keras.preprocessing.image               import load_img, img_to_array, ImageDataGenerator
from scikeras.wrappers                       import KerasClassifier

from keras.layers                            import SeparableConv2D
from keras.layers                            import Dense, Dropout, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, BatchNormalization
from keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, Rescaling, Resizing, Rescaling, RandomZoom, RandomTranslation


from keras.applications.resnet          import ResNet152
from keras.applications.resnet_v2       import ResNet152V2, ResNet101V2
from keras.applications.efficientnet    import EfficientNetB0
from keras.applications.efficientnet_v2 import EfficientNetV2B0

from keras.optimizers import Adam, SGD
from keras.optimizers.schedules import ExponentialDecay

from keras.callbacks  import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from keras.losses     import SparseCategoricalCrossentropy, CategoricalCrossentropy, MSE
from keras.utils      import plot_model

from sklearn.model_selection import cross_val_score

ssl._create_default_https_context = ssl._create_unverified_context

print("TF version:  ", tf.__version__)
print("Hub version: ", hub.__version__)
print("GPU:         ", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

MODEL_NAME    = "Model"
PATH          = "datasets/synthetic_data"
BATCH_SIZE    = 32
IMG_SIZE      = (256, 256)
WIDTH         = 256
HEIGHT        = 256
IMG_SHAPE     = IMG_SIZE + (3,)
EPOCHS        = 20
LEARNING_RATE = 0.003
BETA_1        = 0.9
BETA_2        = 0.999
NR_CLASSES    = 22
NR_NEURONS    = 3 * 1024
WEIGHTS       = 'imagenet'
POOLING       = 'avg'
AUTOTUNE      = tf.data.AUTOTUNE

TRAIN_DIR      = os.path.join(PATH, 'train')
VALIDATION_DIR = os.path.join(PATH, 'valid')
TEST_DIR       = os.path.join(PATH, 'test')

ANNOTATION_PATH = "datasets/annotation_data"

train_annotations_path = os.path.join(ANNOTATION_PATH, 'train_annotations.csv')
valid_annotations_path = os.path.join(ANNOTATION_PATH, 'valid_annotations.csv')

############
# Datasets #
############

train_datagen = ImageDataGenerator(
    featurewise_center            = True,
    featurewise_std_normalization = True,
    rescale                       = 1/255.0,
    shear_range                   = 0.2,
    zoom_range                    = 0.3,
    rotation_range     = 20,
    width_shift_range  = 0.2,
    height_shift_range = 0.2,
    horizontal_flip    = True,
    fill_mode          = 'nearest'
)

valid_datagen = ImageDataGenerator(
    rescale = 1/255.0
)

trainDataset = train_datagen.flow_from_directory(
  directory   = TRAIN_DIR,
  target_size = IMG_SIZE,
  batch_size  = BATCH_SIZE,
  color_mode  = "rgb",
  class_mode  = 'categorical',
  shuffle     = True,
  seed        = 42
)

validDataset = valid_datagen.flow_from_directory(
  directory   = VALIDATION_DIR,
  target_size = IMG_SIZE,
  batch_size  = BATCH_SIZE,
  color_mode  = "rgb",
  class_mode  ='categorical',
  shuffle     = True,
  seed        = 42,
)

testDataset = valid_datagen.flow_from_directory(
  directory   = TEST_DIR,
  target_size = IMG_SIZE,
  batch_size  = BATCH_SIZE,
  color_mode  = "rgb",
  class_mode  = 'categorical',
  shuffle     = False,
  seed        = 42,
)

labels = trainDataset.class_indices
classNames   = []
classIndices = []

for k, v in labels.items():
  classNames.append(k)
  classIndices.append(v)
  print(k,v)

print('Number of classes: ', len(labels))
print('Indices:           ', labels)

###############
# Annotations #
###############

def open_annotations(annotations_path):
  data    = []
  targets = []
  labels  = []
  rows    = open(annotations_path).read().strip().split("\n")

  for row in rows:
    row = row.split(",")

    (filename, letter_class, startX, startY, endX, endY) = row

    image     = load_img(filename, target_size = IMG_SIZE)
    image_arr = img_to_array(image)

    data.append(image_arr)

    startX = round(int(startX) / WIDTH, 2)
    startY = round(int(startY) / HEIGHT, 2)
    endX = round(int(endX) / WIDTH, 2)
    endY = round(int(endY) / HEIGHT, 2)

    targets.append((startX, startY, endX, endY))

    labels.append(letter_class)

  return targets, labels

# train_data, train_targets, train_labels = open_annotations(train_annotations_path)
# valid_data, valid_targets, valid_labels = open_annotations(valid_annotations_path)


############################
# Optimisers
############################

lr_schedule = ExponentialDecay(
  initial_learning_rate = 1e-4,
  decay_steps           = 20000,
  decay_rate            = 0.95
)

ADAM_OPT = Adam(
  learning_rate = lr_schedule,
  beta_1        = BETA_1,
  beta_2        = BETA_2, 
  epsilon       = 1e-08
)

SGD_OPT = SGD(
  learning_rate = LEARNING_RATE, 
  momentum      = 0.9
)

############################
# Pre-trained Models
############################

ResNet152V2Model = ResNet152V2(
  input_shape = IMG_SHAPE,
  include_top = False,
  weights     = "imagenet",
)

ResNet101V2Model = ResNet101V2(
  input_shape = IMG_SHAPE,
  include_top = False,
  weights     = "imagenet",
)

EfficientNetB0 = EfficientNetB0(
  input_shape = IMG_SHAPE,
  include_top = False,
  weights     = "imagenet",
)


# EarlyStopping Callback
earlyStop = EarlyStopping(
  monitor   = 'val_loss',
  mode      = 'max',
  min_delta = 0.001,
  patience  = 15 # it was 10 
)

############################
# ModelCheckpoint Callbacks
############################

checkpoint_filepath_loss = 'models/best-model-{epoch:02d}-{val_loss:.2f}.h5'

checkpointer = ModelCheckpoint(
  filepath          = checkpoint_filepath_loss,
  save_best_only    = True,
  save_weights_only = False,
  mode              = 'max', 
  monitor           = 'val_accuracy',
  verbose           = 1
)

callbacks_list = [earlyStop, checkpointer]


#########################################
# Create a model from a pre trained one #
#########################################

def create_from_trained_model(preTrainedModel):
  for layer in preTrainedModel.layers:
    layer.trainable = False

  base_layers = preTrainedModel.output

  # Add a separable convolutional layer
  base_layers = SeparableConv2D(NR_NEURONS, (3, 3), activation='relu', padding='same')(base_layers)

  # create the classifier branch
  avg                    = GlobalAveragePooling2D()(base_layers)
  mx                     = GlobalMaxPooling2D()(base_layers)
  classifier_layers      = Concatenate()([avg, mx])
  classifier_layers      = BatchNormalization()(classifier_layers)
  classifier_layers      = Dropout(0.5)(classifier_layers)
  classifier_layers      = BatchNormalization()(classifier_layers)
  classifier_layers      = Dropout(0.5)(classifier_layers)
  classifier_predictions = Dense(NR_CLASSES, name='cl_head', activation='softmax')(classifier_layers)

  model = Model(
    inputs  = preTrainedModel.input,
    outputs = [
      classifier_predictions,
      #locator_layers
    ]
  )

  return model

# Create a model from a pre trained one

def create_from_tfhub_model(preTrainedModel):
  efficientnet_v2 = hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/classification/2", trainable = True)
  resnet_v2       = hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_101/classification/5", trainable = True)
  #transformer    = hub.KerasLayer("https://tfhub.dev/sayakpaul/vit_b32_classification/1", trainable = True)

  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape = [256, 256, 3]),
      efficientnet_v2,
      tf.keras.layers.Dropout(rate = 0.3),
      tf.keras.layers.Dense(22, activation='softmax'),
  ])

  return model

model = create_from_trained_model(ResNet152V2Model)


# Build
model.build((None,) + IMG_SIZE + (3,))

# Summary
print(model.summary())


# Compile

#taken from old keras source code
def F1_Score(y_true, y_pred): 
    true_positives      = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives  = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision           = true_positives / (predicted_positives + K.epsilon())
    recall              = true_positives / (possible_positives + K.epsilon())
    f1_val              = 2 * (precision*recall)/(precision+recall+K.epsilon())
    return f1_val

model.compile(
  optimizer = ADAM_OPT,
  loss      = CategoricalCrossentropy(),
  metrics   = ['accuracy', 'Precision', 'Recall', F1_Score]
)

class_weight = {
    0: 1.0,
    1: 1.0,
    2: 2.0,
    3: 1.0,
    4: 1.0,
    5: 2.0,
    6: 1.0
}

history = model.fit(
  trainDataset,
  validation_data  = (validDataset),
  epochs           = EPOCHS,
  callbacks        = [earlyStop, checkpointer]
)

saved_model_path = f"models/model_{MODEL_NAME}"
saved_model_path_2 = f"models/model_{MODEL_NAME}.h5"

# model.save(saved_model_path)
try:
    model.save(saved_model_path)
    model.save(saved_model_path_2)

    print("Model saved successfully.")
except Exception as e:
    print("An error occurred while saving the model:", str(e))
#############################
# Metrics                   #
#############################
acc     = history.history['accuracy']
valAcc  = history.history['val_accuracy']
loss    = history.history['loss']
valLoss = history.history['val_loss']

plt.figure(figsize = (8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc,    label = 'Training Accuracy')
plt.plot(valAcc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss,    label = 'Training Loss')
plt.plot(valLoss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

plt.show()

#############################
# Evaluation                #
#############################
loss, accuracy, precision, recall, f1_score = model.evaluate(testDataset)

print('Model accuracy ---> ', accuracy)
