import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet_v2 import ResNet152V2
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.preprocessing                     import image_dataset_from_directory



#Metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


BATCH_SIZE     = 16
IMG_SIZE       = (256, 256)
IMG_SHAPE      = IMG_SIZE + (3,)
TARGET_SIZE    = (256, 256)
TEST_DATA_PATH =  r"C:\Users\sim-robinnab\Desktop\Phoenician-Project-Robin_Nabhan\datasets\synthetic_data\test"



test_dataset = image_dataset_from_directory(TEST_DATA_PATH,
  shuffle    = True,
  batch_size = BATCH_SIZE,
  image_size = IMG_SIZE
)

test_datagen = ImageDataGenerator(
  rescale = 1/255.0
)

testDataset = test_datagen.flow_from_directory(
  directory   = TEST_DATA_PATH,
  target_size = IMG_SIZE,
  batch_size  = BATCH_SIZE,
  color_mode  = 'rgb',
  class_mode  = None,
  shuffle     = False,
  seed        = 42,
)


classNames = test_dataset.class_names
print(classNames)
NR_CLASSES = len(classNames)


def make_prediction(model):
  imageBatch, labelBatch = test_dataset.as_numpy_iterator().next()
  predictedBatch         = model.predict(imageBatch)
  predictedId            = np.argmax(predictedBatch, axis = -1)

  plt.figure(figsize = (10, 10))

  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)

    plt.imshow(imageBatch[i].astype("uint8"))
    plt.title(classNames[predictedId[i]])
    plt.axis("off")

  plt.tight_layout()
  plt.show()

def F1_Score(y_true, y_pred):
    true_positives      = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives  = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision           = true_positives / (predicted_positives + K.epsilon())
    recall              = true_positives / (possible_positives + K.epsilon())
    f1_val              = 2 * (precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def loading_model(MODEL_PATH):
  print('')
  print('--> Loading model ', MODEL_PATH)

  model = load_model(MODEL_PATH, custom_objects = { 'KerasLayer': hub.KerasLayer, 'F1_Score': F1_Score })

  test_loss, test_acc, precision, recall, f1_score  = model.evaluate(testDataset)

  print('Model {}  Test loss: {} Test Acc: {}'.format(MODEL_PATH, test_loss, test_acc))

  #make_prediction(model)

  return model


print('Loading models')

# efficientnetv2 = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
#     include_top=True,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation='softmax',
#     include_preprocessing=True
# )
# resnet152v2 = tf.keras.applications.efficientnet_v2.EfficientNetV2B2(
#     include_top=True,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation='softmax',
#     include_preprocessing=True
# )
# EFFICIENTNETV2_PATH =  os.path.join(basedir, "../models/efficientnetv2B0-0.999.h5")
# RESNET152v2_PATH    = os.path.join(basedir, "../models/model_Model")

# resnet152v2    = loading_model(RESNET152v2_PATH)
# # efficientnetv2 = loading_model(EFFICIENTNETV2_PATH)

# deepLModels = [
#   { 'id': 1, 'name': 'ResNet152v2',     'model': resnet152v2 },
#   { 'id': 2, 'name': 'EfficientNetB7',  'model': efficientnetv2 },
#   { 'id': 3, 'name': 'EfficientNetV2L', 'model': efficientnetv2 }
# ]

# defaultModel = deepLModels[0]

def getModelById(modelId):
  model = ''

  for deepLModel in deepLModels:
    if modelId == deepLModel['id']:
      model = deepLModel

  if not model:
    model = defaultModel

  print(model['name'])

  return model['model']


def getModelNameById(modelId):
  model = ''

  for deepLModel in deepLModels:
    if modelId == deepLModel['id']:
      model = deepLModel

  return model['name']

def loadImage(imgPath):
  image     = load_img(imgPath, target_size = TARGET_SIZE)
  input_arr = img_to_array(image)
  input_arr = np.array([input_arr])
  input_arr = input_arr.astype('float32') / 255.

  return input_arr

#############################
# Predict an image          #
#############################
def predictByCompactBilinearPooling(imgPath):
  image          = loadImage(imgPath)
  returnObjects  = []

  for deepLModel in deepLModels:
    predictions    = deepLModel['model'].predict(image)
    maxPredIndex   = np.argmax(predictions[0])
    maxPredPercent = round((100 * np.max(predictions[0])), 2)

    for i in range(len(classNames)):
      if i == maxPredIndex:
        predictionObj = {
          'letter':     classNames[i],
          'prediction': maxPredPercent,
          'model':      deepLModel['name']
        }

        returnObjects.append(predictionObj)
        break

  print('')
  print('-------------------------------------')
  for obj in returnObjects:
    print('Prediction -> Letter: {} Accuracy: {} Model: {}'.format(obj['letter'], obj['prediction'], obj['model']))
  print('-------------------------------------')
  print('')

  return returnObjects

#############################
# Predict an image #
#############################
def predictByModel(imgPath, modelId):
  image         = loadImage(imgPath)
  model         = getModelById(modelId)

  predictions   = model.predict(image)
  top_indices   = np.argsort(predictions)[0, ::-1][:3]
  top_indice    = np.argmax(predictions[0])
  top_percents  = np.sort(predictions)[0, ::-1][:3]
  percent       = round((100 * np.max(predictions[0])), 3)
  returnObjects = []

  print('classNames ->      ', len(classNames))
  print('top_indices ->      ', len(top_indices))

  for i in range(len(classNames)):
    for j in range(len(top_indices)):
      classNames[i]
      print('i ->      ', i)
      print('top_indices[j] ->      ', top_indices[j])
      if i == top_indices[j]:
        predictionObj = {
          'letter':     classNames[i],
          'prediction': round((100 * top_percents[j]), 2)
        }
        returnObjects.append(predictionObj)

  print('-------------------------------------')
  # print('Shape ->        ', predictions.shape)
  # print('Predictions ->  ', predictions[0])
  print('Top Indices ->  ', top_indices)
  print('Top Percents -> ', top_percents)
  print('Index ->        ', top_indice)
  print('Percent ->      ', percent)
  print('Letters ->      ', returnObjects)
  print('-------------------------------------')
  print('')

  return returnObjects

# Define the custom F1_Score function
def F1_Score(y_true, y_pred):
    true_positives      = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives  = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision           = true_positives / (predicted_positives + K.epsilon())
    recall              = true_positives / (possible_positives + K.epsilon())
    f1_val              = 2 * (precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# Define constants
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
MODEL_PATH = r"C:\Users\sim-robinnab\Desktop\Phoenician-Project-Robin_Nabhan\models\best-model-16-0.23.h5"  # Replace with your model's path
TARGET_SIZE = (256, 256)
TEST_DATA_PATH = r'C:\Users\sim-robinnab\Desktop\Phoenician-Project-Robin_Nabhan\datasets\synthetic_data\test'

# Load the model
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
    'GlobalAveragePooling2D': GlobalAveragePooling2D,
    'F1_Score': F1_Score  # Pass the F1_Score function as a custom object
})

# Preprocess and augment the image
img_path = r"C:\Users\sim-robinnab\Desktop\Phoenician-Project-Robin_Nabhan\datasets\real_image_dataset\Qof\Qof2.png"  # Replace with your image path
image = load_img(img_path, target_size=TARGET_SIZE)
image_array = img_to_array(image)
image_array = image_array[np.newaxis, :]  # Add batch dimension
datagen = ImageDataGenerator(rescale=1. / 255.0)
datagen.fit(image_array)



# Make prediction
predictions = model.predict(datagen.flow(image_array, batch_size=BATCH_SIZE))

# Process and interpret predictions
predicted_index = np.argmax(predictions[0])


predicted_class = classNames[predicted_index]
prediction_confidence = predictions[0][predicted_index]

print(f"Predicted class: {predicted_class} ({prediction_confidence:.2f} confidence)")

# Visualize the image (optional)
plt.imshow(image)
plt.title(f"Predicted class: {predicted_class}")
plt.show()
