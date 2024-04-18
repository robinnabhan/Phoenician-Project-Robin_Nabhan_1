import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
import tensorflow        as tf

from tensorflow.keras.preprocessing       import image_dataset_from_directory, image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.models              import load_model
from tensorflow.keras.models              import Model

from IPython.display import Image, display

RESNET101_PATH        = "models/model_ResNet101_0.964.h5"
RESNET152_PATH        = "models/model_ResNet152_0.929.h5"
VGG19_PATH            = "models/model_VGG19_0.893.h5"
EFFICIENT_NET_B7_PATH = "models/model_EfficientNetB7_0.786.h5"

#resNet101Model      = load_model(RESNET101_PATH)
resNet152Model      = load_model(RESNET152_PATH)
#vgg19Model          = load_model(VGG19_PATH)
#efficientNetB7Model = load_model(EFFICIENT_NET_B7_PATH)

deepLModels = [
  { 'id': 1, 'name': 'ResNet101',      'model': "resNet101Model" },
  { 'id': 2, 'name': 'ResNet152',      'model': resNet152Model },
  { 'id': 3, 'name': 'VGG19',          'model': "vgg19Model" },
  { 'id': 4, 'name': 'EfficientNetB7', 'model': "efficientNetB7Model" }
]

defaultModel = deepLModels[2]

PATH           = "datasets/panamuwa"
BATCH_SIZE     = 32
IMG_SIZE       = (224, 224)
TARGET_SIZE    = (224, 224, 3)
TRAIN_DIR      = os.path.join(PATH, 'train')
VALIDATION_DIR = os.path.join(PATH, 'valid')
TEST_DIR       = os.path.join(PATH, 'test')
basedir        = os.path.abspath(os.path.dirname(__file__))

# Datasets
trainDataset = image_dataset_from_directory(TRAIN_DIR,
  validation_split = 0.2,
  subset           = "training",
  seed             = 1337,
  image_size       = IMG_SIZE,
  batch_size       = BATCH_SIZE,
  shuffle          = True
)

validationDataset = image_dataset_from_directory(VALIDATION_DIR,
  validation_split = 0.2,
  subset           = "validation",
  seed             = 1337,
  shuffle          = True,
  image_size       = IMG_SIZE,
  batch_size       = BATCH_SIZE
)

testDataset = image_dataset_from_directory(TEST_DIR, 
  shuffle    = True, 
  batch_size = BATCH_SIZE, 
  image_size = IMG_SIZE
)

classNames = trainDataset.class_names 

def getModelById(modelId):
  model = ''

  for deepLModel in deepLModels:
    if modelId == deepLModel['id']:
      model = deepLModel
  
  if not model:
    model = defaultModel
  
  print(model['name'])

  return model['model']
  
def loadImage(imgPath):
  imageRaw        = load_img(imgPath, target_size = TARGET_SIZE)
  image           = img_to_array(imageRaw)
  predictionImage = np.array(image)
  predictionImage = np.expand_dims(image, 0)
  return predictionImage

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
  # First, we create a model that maps the input image to the activations
  # of the last conv layer as well as the output predictions
  
  grad_model = Model(
    [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
  )

  # Then, we compute the gradient of the top predicted class for our input image
  # with respect to the activations of the last conv layer
  with tf.GradientTape() as tape:
    last_conv_layer_output, preds = grad_model(img_array)
    
    if pred_index is None:
      pred_index = tf.argmax(preds[0])

    class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

#############################
# Predict an image #
#############################
def makeHeatMap(imgPath, modelId, lastConvLayer):  
  image = loadImage(imgPath)
  model = getModelById(modelId)

  # Remove last layer's softmax
  model.layers[-1].activation = None

  model.summary()

  predictions = model.predict(image)

  # Generate class activation heatmap
  heatmap = make_gradcam_heatmap(image, model, lastConvLayer)

  # Display heatmap
  #plt.matshow(heatmap)
  #plt.show()

  return heatmap

def saveGradCam(imgPath, heatmap, cam_path = "cam.jpg", alpha=0.4):
  # Load the original image
  img = image.load_img(imgPath)
  img = img_to_array(img)

  # Rescale heatmap to a range 0-255
  heatmap = np.uint8(255 * heatmap)

  # Use jet colormap to colorize heatmap
  jet = cm.get_cmap("jet")

  # Use RGB values of the colormap
  jet_colors = jet(np.arange(256))[:, :3]
  jet_heatmap = jet_colors[heatmap]

  # Create an image with RGB colorized heatmap
  jet_heatmap = array_to_img(jet_heatmap)
  jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
  jet_heatmap = img_to_array(jet_heatmap)

  # Superimpose the heatmap on original image
  superimposed_img = jet_heatmap * alpha + img
  superimposed_img = array_to_img(superimposed_img)

  # Save the superimposed image
  superimposed_img.save(cam_path)

  # Display Grad CAM
  display(Image(cam_path))

def test():
  filePath1     = 'client/static/uploads/Panamuwa_1_1.png'
  filePath2     = 'client/static/uploads/Panamuwa_1_2.png'
  camPath1 = "cam1.jpg"
  camPath2 = "cam2.jpg"

  modelId       = 2
  lastConvLayer = 'conv5_block3_out'

  heatmap1 = makeHeatMap(filePath1, modelId, lastConvLayer)
  heatmap2 = makeHeatMap(filePath2, modelId, lastConvLayer)

  saveGradCam(filePath1, heatmap1, camPath1)

  saveGradCam(filePath2, heatmap2, camPath2)

test()