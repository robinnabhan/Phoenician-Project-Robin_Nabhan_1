import cv2
import numpy as np

from PIL             import Image, ImageEnhance
from sklearn.cluster import MiniBatchKMeans

def contrast(image):
    if np.random.uniform(0, 1) < 0.8:
        factor = np.random.uniform(1, 2)
    else:
        factor = np.random.uniform(0.5, 1)

    enhancer = ImageEnhance.Contrast(image)
    image    = enhancer.enhance(factor)

    return image

def brightness(image):
    brightness_factor = np.random.uniform(0.4,1.1)
    enhancer          = ImageEnhance.Brightness(image)
    image             = enhancer.enhance(brightness_factor)

    return image

def sharpness(img):
    if np.random.uniform(0,1)<0.5:
        # increase sharpness
        factor = np.random.uniform(0.1,1)
    else:
        # decrease sharpness
        factor = np.random.uniform(1,10)

    enhancer = ImageEnhance.Sharpness(img)

    img  = enhancer.enhance(factor)

    return img


def s_and_p(img):
    image_arr = np.asarray(img)

    amount = np.random.uniform(0.001, 0.01)
    
    # add some s&p
    s_vs_p = 0.5
    out = np.copy(image_arr)
    # Salt mode
    num_salt = np.ceil(amount * image_arr.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in image_arr.shape]
    out[coords] = 1

    #pepper
    num_pepper = np.ceil(amount* image_arr.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in image_arr.shape]
    out[coords] = 0
    
    img = Image.fromarray(out)

    return img

def scale(img):

    image_arr = np.asarray(img)

    f = np.random.uniform(0.5,1.5)

    shape_OG = image_arr.shape

    res = cv2.resize(image_arr,None,fx=f, fy=f, interpolation = cv2.INTER_CUBIC)

    res = cv2.resize(res,None,fx=1.0/f, fy=1.0/f, interpolation = cv2.INTER_CUBIC)
    
    img = Image.fromarray(res)

    return img


def quantize(img):
    image_arr = np.asarray(img)

    image_arr = cv2.cvtColor(image_arr, cv2.COLOR_GRAY2RGB)
    
    (h, w) = image_arr.shape[:2]

    image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2LAB)
    image_arr = image_arr.reshape((image_arr.shape[0] * image_arr.shape[1], 3))

    clt = MiniBatchKMeans(n_clusters = 2)
    labels = clt.fit_predict(image_arr)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    quant = quant.reshape((h, w, 3))
    image_arr = image_arr.reshape((h, w, 3))

    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image_arr = cv2.cvtColor(image_arr, cv2.COLOR_LAB2BGR)
    
    image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)

    img = Image.fromarray(image_arr)

    return img

def invert(img):
    image_arr = np.asarray(img)

    im_inv = cv2.bitwise_not(image_arr)
    
    img = Image.fromarray(im_inv)

    return img

def darken(img):
    image_arr = np.asarray(img)

    image_arr = cv2.subtract(image_arr, np.random.uniform(0, 50))

    img = Image.fromarray(image_arr)

    return img

def degrade_img(img):

    # if np.random.uniform(0,1) < 0.1:
    #     img = s_and_p(img)

    if np.random.uniform(0,1) < 0.5:
        img = scale(img)

    if np.random.uniform(0,1) < 0.5:
        img = brightness(img)

    if np.random.uniform(0,1) < 0.7:
        img = contrast(img)

    if np.random.uniform(0,1) < 0.5:
        img = sharpness(img)

    # if np.random.uniform(0,1) < 0.4:
    #     img = quantize(img)

    if np.random.uniform(0,1) < 0.3:
        img = darken(img)

    image_arr = np.asarray(img)
    
    # image_arr = cv2.resize(image_arr, (224,224))

    img = Image.fromarray(image_arr)

    return img
