
import io
import os
import itertools
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
import tensorflow        as tf
import seaborn           as sns
import tensorflow_hub    as hub

from datetime import datetime

from scikitplot.metrics import plot_roc, plot_precision_recall
from sklearn.metrics    import confusion_matrix

from keras.preprocessing                     import image_dataset_from_directory, image
from keras.preprocessing.image               import load_img, img_to_array, array_to_img, ImageDataGenerator
from keras.models                            import Model, Sequential, load_model
from keras.layers.experimental.preprocessing import Resizing, Rescaling

RESNET152v2_PATH = "models/efficientnetv2B0.h5"

BATCH_SIZE  = 1
IMG_SIZE    = (224, 224)
TARGET_SIZE = (224, 224, 3)
TEST_DIR    = "datasets/real_image_data"
basedir     = os.path.abspath(os.path.dirname(__file__))

# Create log directory
logdir         = "logs/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')

test_datagen = ImageDataGenerator(
  rescale = 1/255.0
)

testDataset = test_datagen.flow_from_directory(
  directory   = TEST_DIR,
  target_size = IMG_SIZE,
  batch_size  = BATCH_SIZE,
  color_mode  = 'rgb',
  class_mode  = None,
  shuffle     = False,
  seed        = 42,
)

labels       = testDataset.class_indices
classNames   = []
classIndices = []
testClassIndices = testDataset.classes

for k, v in labels.items():
  classNames.append(k)
  classIndices.append(v)

print('Number of classes: ', len(labels))
print('classNames:        ', classNames)
print('classIndices:      ', classIndices)
print('Classes:           ', testClassIndices)
print('testClassIndices Len: ', len(testClassIndices))

def loading_model(MODEL_PATH):
  '''
  Loads the model from the path
  '''
  print('Loading model ', MODEL_PATH)

  model               = load_model(MODEL_PATH, custom_objects={'KerasLayer':hub.KerasLayer})
  test_loss, test_acc = model.evaluate(testDataset)

  print('Model {}  Test loss: {} Test Acc: {}'.format(MODEL_PATH, test_loss, test_acc))

  return model

def loadImage(imgPath):
  '''
  Loads the image from the path
  '''

  image = load_img(imgPath, target_size = (224, 224))
  image = img_to_array(image)
  image = np.array([image])
  image = image / 255

  return image

def test_prediction(model, imgPath):
  '''
  Predicts the image from the path
  '''

  image         = loadImage(imgPath)
  predictions   = model.predict(image)

  top_indices   = np.argsort(predictions)[0, ::-1][:3]
  top_indice    = np.argmax(predictions[0])
  top_percents  = np.sort(predictions)[0, ::-1][:3]
  percent       = round((100 * np.max(predictions[0])), 3)
  returnObjects = []

  for j in range(len(top_indices)):
    for i in range(len(classNames)):
      if i == top_indices[j]:
        predictionObj = {
          'letter':     classNames[i],
          'prediction': round((100 * top_percents[j]), 2)
        }
        returnObjects.append(predictionObj)

  print('-------------------------------------')
  #print('Predictions ->  ', predictions[0])
  print('Img Path ->     ', imgPath)
  print('Top Indices ->  ', top_indices)
  print('Top Percents -> ', top_percents)
  print('Top Index ->    ', top_indice)
  print('Top Class ->    ', classNames[top_indice])
  print('Top Class % ->  ', percent)
  for i in range(len(returnObjects)):
    print('Letters ->      ', returnObjects[i])

  print('-------------------------------------')
  print('')

def plot_image_confusion_matrix(model, image_path):
  """
  This function prints and plots the confusion matrix.
  """

  sample_ds  = loadImage(image_path)
  prediction = model.predict(sample_ds)

  plt.bar(classNames, tf.nn.softmax(prediction[0]))
  plt.title('Predictions')
  plt.show()

def plot_confusion_matrix_2(model, test_ds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    y_pred = np.argmax(model.predict(test_ds), axis = 1)

    y_true = testClassIndices

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize = (10, 8))
    
    sns.heatmap(
      confusion_mtx,
      xticklabels = classNames,
      yticklabels = classNames,
      annot = True,
      fmt = 'g'
    )

    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


def plot_confusion_matrix(cm, class_names):
  '''
  This function prints and plots the confusion matrix.
  '''

  figure = plt.figure(figsize=(8, 8))

  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()

  tick_marks = np.arange(len(class_names))

  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)
    
  # Normalize the confusion matrix.
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
    
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

def calc_confusion_matrix(model, test_ds):
  '''
  Predict the values from the test dataset
  Calculates the confusion matrix
  '''
  test_pred_raw = model.predict(test_ds)
  test_pred     = np.argmax(test_pred_raw, axis = 1)  
  cm            = confusion_matrix(testClassIndices, test_pred)
  figure        = plot_confusion_matrix(cm, class_names=classNames)

def calc_roc_curve(model, test_ds):
  probs = model.predict(test_ds)

  plot_roc(testClassIndices, probs)

  plot_precision_recall(testClassIndices, probs)

  plt.show()

def main():
  resNet152v2    = loading_model(RESNET152v2_PATH)

  files = [f for f in os.listdir(TEST_DIR + '/Shin') if f.endswith('.png')]
  for letterfile in files:
    test_prediction(resNet152v2, TEST_DIR + '/Shin/' + letterfile)

  #test_prediction(resNet152v2, TEST_DIR + '/Shin/imageedit_131_8818355424.png')

  #plot_image_confusion_matrix(resNet152v2, TEST_DIR + '/Shin/imageedit_131_8818355424.png')


  calc_confusion_matrix(resNet152v2, testDataset)

  # calc_roc_curve(resNet152v2, testDataset)

  # plot_confusion_matrix_2(resNet152v2, testDataset)

if __name__ == "__main__":
  main()
