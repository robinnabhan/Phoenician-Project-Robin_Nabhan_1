import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
import tensorflow        as tf

from tensorflow.keras.preprocessing                     import image_dataset_from_directory, image
from tensorflow.keras.preprocessing.image               import load_img, img_to_array, array_to_img
from tensorflow.keras.models                            import Model, Sequential, load_model
from tensorflow.keras.layers.experimental.preprocessing import Resizing, Rescaling

RESNET50v2_PATH     = "models/model-resNet50v2-0.931.h5"
RESNET152v2_PATH    = "models/model-resNet152v2-0.972.h5"


BATCH_SIZE  = 16
IMG_SIZE    = (224, 224)
TARGET_SIZE = (224, 224, 3)

PATH     = "datasets/synthetic_data"
TEST_DIR = "datasets/real_image_data"
basedir  = os.path.abspath(os.path.dirname(__file__))

testDataset = image_dataset_from_directory(TEST_DIR,
  shuffle    = True, 
  batch_size = BATCH_SIZE, 
  image_size = IMG_SIZE
)

with tf.device('/cpu:0'):
  data_augmentation = Sequential([
    #Resizing(224, 224),
    Rescaling(1. / 255)
  ])

aug_test_dataset = testDataset.map(lambda x, y: (data_augmentation(x, training = True), y))

classNames = testDataset.class_names

def loadImage(imgPath):
  image = load_img(imgPath, target_size = (224, 224))
  
  image = img_to_array(image)
  
  image = np.array([image])
  
  image = image / 255

  return image

# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
#   # First, we create a model that maps the input image to the activations
#   # of the last conv layer as well as the output predictions
  
#   grad_model = Model(
#     [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
#   )

#   # Then, we compute the gradient of the top predicted class for our input image
#   # with respect to the activations of the last conv layer
#   with tf.GradientTape() as tape:
#     last_conv_layer_output, preds = grad_model(img_array)
    
#     if pred_index is None:
#       pred_index = tf.argmax(preds[0])

#     class_channel = preds[:, pred_index]

#     # This is the gradient of the output neuron (top predicted or chosen)
#     # with regard to the output feature map of the last conv layer
#     grads = tape.gradient(class_channel, last_conv_layer_output)

#     # This is a vector where each entry is the mean intensity of the gradient
#     # over a specific feature map channel
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     # We multiply each channel in the feature map array
#     # by "how important this channel is" with regard to the top predicted class
#     # then sum all the channels to obtain the heatmap class activation
#     last_conv_layer_output = last_conv_layer_output[0]
#     heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)

#     # For visualization purpose, we will also normalize the heatmap between 0 & 1
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

#     return heatmap.numpy()

#############################
# Predict an image #
#############################

# def makeHeatMap(imgPath, lastConvLayer):  
#   image = loadImage(imgPath)

#   # Remove last layer's softmax
#   model.layers[-1].activation = None

#   model.summary()

#   # Generate class activation heatmap
#   heatmap = make_gradcam_heatmap(image, model, lastConvLayer)

#   return heatmap

# def saveGradCam(imgPath, heatmap, cam_path, alpha=0.4):
#   img = image.load_img(imgPath)
#   img = img_to_array(img)

#   # Rescale heatmap to a range 0-255
#   heatmap = np.uint8(255 * heatmap)

#   # Use jet colormap to colorize heatmap
#   jet = cm.get_cmap("jet")

#   # Use RGB values of the colormap
#   jet_colors = jet(np.arange(256))[:, :3]
#   jet_heatmap = jet_colors[heatmap]

#   # Create an image with RGB colorized heatmap
#   jet_heatmap = array_to_img(jet_heatmap)
#   jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
#   jet_heatmap = img_to_array(jet_heatmap)

#   # Superimpose the heatmap on original image
#   superimposed_img = jet_heatmap * alpha + img
#   superimposed_img = array_to_img(superimposed_img)

#   # Save the superimposed image
#   superimposed_img.save(cam_path)


def loading_model(MODEL_PATH):
  print('')
  print('--> Loading model ', MODEL_PATH)

  model = load_model(MODEL_PATH)

  test_loss, test_acc = model.evaluate(aug_test_dataset)

  print('Model {}  Test loss: {} Test Acc: {}'.format(MODEL_PATH, test_loss, test_acc))

  return model

#resNet50v2    = loading_model(RESNET50v2_PATH)
resNet152v2    = loading_model(RESNET152v2_PATH)

def test_prediction(model, imgPath):
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
  print('Percent ->      ', percent)
  print('Letters ->      ', returnObjects)
  print('-------------------------------------')
  print('')



def main():
  files = [f for f in os.listdir(TEST_DIR + '/Waw') if f.endswith('.png')]

  for letterfile in files:
    test_prediction(resNet152v2, TEST_DIR + '/Waw/' + letterfile)

if __name__ == "__main__":
  main()
