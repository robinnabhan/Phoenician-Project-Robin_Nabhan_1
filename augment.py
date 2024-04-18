import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.filters       import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from PIL                         import Image

def elastic_transform(image, alpha, sigma, random_state = None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    new_random_state = (random_state.rand(*shape) * 2 - 1)

    dx = gaussian_filter(new_random_state, sigma, mode = "constant", cval = 0) * alpha
    dy = gaussian_filter(new_random_state, sigma, mode = "constant", cval = 0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(
        np.arange(shape[0]), 
        np.arange(shape[1]), 
        np.arange(shape[2])
    )
    
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=np.random.randint(1,5), mode='reflect')

    return distored_image.reshape(image.shape)


def distort(img):
    '''
    Distort image randomly.
    '''
    sigma_alpha = [
        (np.random.randint(9,11), np.random.randint(2,4)),
        (np.random.randint(80,100), np.random.randint(4,5)),
        (np.random.randint(150, 300), np.random.randint(5, 6)),
        (np.random.randint(800, 1200), np.random.randint(8,10)),
        (np.random.randint(1500, 2000), np.random.randint(10, 15)),
        (np.random.randint(5000, 8000), np.random.randint(15, 25)),
        # (np.random.randint(10000, 15000), np.random.randint(20, 25)),
        # (np.random.randint(45000, 55000), np.random.randint(30, 35)),
    ]

    choice = np.random.randint(len(sigma_alpha))

    sigma_alpha_chosen = sigma_alpha[choice]

    return elastic_transform(img, sigma_alpha_chosen[0], sigma_alpha_chosen[1])


def rotate_letter(image):
    angle = np.random.randint(0, 20)
    
    image = image.rotate(angle, resample = Image.BICUBIC, expand = True)

    return image

def resize_letter(image):
    zoom  = np.random.random() * 0.4 + 0.5 # Zoom in range [0.5,1.2)

    new_size = (int(image.size[0] * zoom), int(image.size[1] * zoom))

    image = image.resize(new_size, resample=Image.BICUBIC)

    return image

def rotate(img, obj=None):
    '''
    Rotate image between 0-360 degrees randomly.
    '''

    rows,cols,_ = img.shape

    angle = np.random.randint(0,20)

    col=(float(img[0][0][0]),float(img[0][0][1]),float(img[0][0][2]))

    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)

    if obj == "mol": 
        dst = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_DEFAULT)

    if obj == "bkg": 
        dst = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REFLECT)

    return dst

def resize(img):
    '''
    Resize image random from between (200-300, 200-300) with a random choice of interpolation
    and then resize back to 256x256.
    '''
    interpolations = [cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

    img = cv2.resize(img, (np.random.randint(200,300), np.random.randint(200,300)), interpolation = np.random.choice(interpolations))
    img = cv2.resize(img, (256,256), interpolation = np.random.choice(interpolations))

    return img


def gaussian_blur(image):
    """[summary]
        Apply Gaussian blur
    Args:
        img ([type]): [description]
    """

    # sigmaX is Gaussian Kernel standard deviation
    # ksize is kernel size
    image = cv2.GaussianBlur(src = image, ksize=(5, 5), sigmaX = 0, sigmaY = 0)

    return image

def blur(img):
    '''
    Blur image randomly 1-2.
    '''

    n      = np.random.randint(1, 3)
    kernel = np.ones((n, n), np.float32) / n ** 2
    img    = cv2.filter2D(img, -1, kernel)

    return img

def erode(img):
    n      = np.random.randint(1,3)
    kernel = np.ones((n, n), np.float32) / n ** 2
    img    = cv2.erode(img, kernel, iterations = 1)

    return img

def dilate(img):
    n      = 2
    kernel = np.ones((n,n),np.float32)/n**2
    img    = cv2.dilate(img, kernel, iterations=1)

    return img


def aspect_ratio(img, obj=None):
    if obj == "mol":
        n = 50
        image = cv2.copyMakeBorder(img, np.random.randint(0,50),
                       np.random.randint(0,50), np.random.randint(0,50),
                       np.random.randint(0,50), cv2.BORDER_CONSTANT,
                       value=[255,255,255])
    elif obj == "bkg":
        n = 100
        image = cv2.copyMakeBorder(img, np.random.randint(0,50),
                               np.random.randint(0,50), np.random.randint(0,50),
                               np.random.randint(0,50), cv2.BORDER_REFLECT)
    
    image = cv2.resize(image, (256,256))
    
    return image

def crop_bkg(img):
    x, y, _ =  img.shape
    n1      = np.random.randint(0,50)
    n2      = np.random.randint(n1+150, 256)
    n3      = np.random.randint(0,50)
    n4      = np.random.randint(n3+150, 256)
    cropped = img[n1:n2,n3:n4]
    cropped = cv2.resize(cropped, (256,256))
    
    return cropped

def translate_bkg(img):
    # translate
    rows, cols  = img.shape[:2]
    r1 = np.random.uniform(-100, 100)
    r2 = np.random.uniform(-100, 100)
    M = np.float32([[1,0,r1], [0,1,r2]])
    dst = cv2.warpAffine(img, M, (cols, rows), borderMode = cv2.BORDER_REFLECT, borderValue = [255,255,255])

    return dst

def translate_mol(img):
    img = cv2.resize(img, (350, 350))

    mol = crop_mol(img)
    rows_m, cols_m = mol.shape
    
    y_max = 350-rows_m
    x_max = 350-cols_m
    if y_max > 0 and x_max > 0:
        y_offset=np.random.randint(0,350-rows_m)
        x_offset=np.random.randint(0,350-cols_m)
    else:
        y_offset=0
        x_offset=0

    white = np.zeros((350, 350), np.uint8)
    white[:] = (350)
    white[y_offset:y_offset+rows_m, x_offset:x_offset+cols_m] = mol

    return white

def add_border(img):
    if np.random.uniform(0,1)<0.2:
        col = [np.random.uniform(0,255), np.random.uniform(0,255),np.random.uniform(0,255)]
        border = cv2.copyMakeBorder(img, top=np.random.randint(0,40), bottom=np.random.randint(0,40),
                                left=np.random.randint(0,40), right=np.random.randint(0,40),
                                borderType= cv2.BORDER_CONSTANT, value=col)
    else:
        border = cv2.copyMakeBorder(img, top=np.random.randint(0,40), bottom=np.random.randint(0,40),
                                left=np.random.randint(0,40), right=np.random.randint(0,40),
                                borderType= cv2.BORDER_REFLECT)
    border = cv2.resize(border, (256,256))

    return border

def affine(img, obj=None):
    rows, cols,_ = img.shape
    n = 20
    pts1 = np.float32([[5, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[5 + np.random.randint(-n, n), 50 + np.random.randint(-n, n)],
                   [200 + np.random.randint(-n, n), 50 + np.random.randint(-n, n)],
                   [50 + np.random.randint(-n, n), 200 + np.random.randint(-n, n)]])

    M = cv2.getAffineTransform(pts1, pts2)

    if obj == "mol":
        skewed = cv2.warpAffine(img, M, (cols, rows), borderValue=[255,255,255])
    elif obj == "bkg":
        skewed = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    skewed = cv2.resize(skewed, (256,256))
    return skewed

def flip_v(img):
    return cv2.flip( img, 0)

def flip_h(img):
    return cv2.flip(img, 1)

# function gets bounding box of molecule with a pixel either side
def get_bounding_box(img):
    x_min = img.shape[0]
    x_max = 0
    y_min = img.shape[1]
    y_max = 0

    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] < 250:
                if i < x_min:
                    x_min = i
                if i > x_max:
                    x_max = i
                if j < y_min:
                    y_min = j
                if j > y_max:
                    y_max = j

    return x_min-1, x_max+1, y_min-1, y_max+1


def crop_mol(img):
    # crop to bounding box
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_min, x_max, y_min, y_max = get_bounding_box(img)
    img_crop = img[x_min:x_max,y_min:y_max]

    return img_crop


def augment_letter(image):
    """[summary]

    Args:
        img ([type]): [description]

    Returns:
        [type]: [description]
    """

    image = rotate_letter(image)

    image = resize_letter(image)

    image_arr = np.asarray(image)

    image_arr = gaussian_blur(image_arr)

    if np.random.uniform(0,1) < 0.3:
        image_arr = erode(image_arr)

    if np.random.uniform(0,1) < 0.3:
        image_arr = dilate(image_arr)

    image_arr = aspect_ratio(image_arr, "mol")

    if np.random.uniform(0,1) < 0.7:
        image_arr = affine(image_arr, "mol")

    # # flip_v    
    # if np.random.uniform(0,1) < 0.5:
    #     image_arr = flip_v(image_arr)
    
    # # flip_h    
    # if np.random.uniform(0,1) < 0.5:
    #     image_arr = flip_h(image_arr)
    
    if np.random.uniform(0,1) < 0.7:
        image_arr = distort(image_arr)

    # if np.random.uniform(0,1) < 0.3:
    #   image_arr = translate_mol(image_arr)

    new_image = Image.fromarray(image_arr)

    return new_image

def augment_bkg(image):
    """[summary]

    Args:
        img ([type]): [description]

    Returns:
        [type]: [description]
    """

    image_arr = np.asarray(image)

    image_arr = rotate(image_arr, "bkg")

    image_arr = gaussian_blur(image_arr)

    image_arr = erode(image_arr)

    if np.random.uniform(0,1) < 0.3:
        image_arr = dilate(image_arr)

    if np.random.uniform(0,1) < 0.3:
        image_arr = translate_bkg(image_arr)

    if np.random.uniform(0,1) < 0.7:
        image_arr = affine(image_arr, "bkg")

    # if np.random.uniform(0,1) < 0.5:
    #     image_arr = flip_v(image_arr)

    # if np.random.uniform(0,1) < 0.5:
    #     image_arr = flip_h(image_arr)

    if np.random.uniform(0,1) < 0.8:
        image_arr = distort(image_arr)

    image_arr = crop_bkg(image_arr)

    new_image = Image.fromarray(image_arr)

    return new_image
