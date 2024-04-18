import os
import random
import pathlib
import csv
import cv2
import numpy             as np
import matplotlib.pyplot as plt

import pascal 
from PIL                 import Image, ImageColor, ImageFilter, ImageDraw, ImageEnhance
from matplotlib.patches  import Rectangle
from augment             import augment_letter, augment_bkg
from degrade             import degrade_img

BASE_DIR = os.path.dirname(__file__)

LETTERS_PATH           = "./images/letters-phoenician/" #it was images/letters 
LETTERS_VARIATION_PATH = "./images/letters-phoenician/"
#LETTERS_VARIATION_PATH = "./images/Nabatean_script/"
BACKGROUNDS_PATH       = "./images/textures/basalt/"

# Folders for training, testing, and validation subsets
DATASET_PATH     = './datasets/synthetic_data/'

# Folders for training, testing, and validation subsets
dir_data  = pathlib.Path.cwd().joinpath('datasets/synthetic_data')
dir_train = dir_data.joinpath('train')
dir_valid = dir_data.joinpath('valid')
dir_test  = dir_data.joinpath('test')

# Output the annotations csv
train_annotations_csv_path = os.path.join(dir_train, 'annotations.csv')

# Output the annotations csv
valid_annotations_csv_path = os.path.join(dir_valid, 'annotations.csv')

# Train/Test/Validation split config
PCT_TRAIN = 0.8
PCT_VALID = 0.1
PCT_TEST  = 0.1

TEXTURE_LETTERS = [
  '#3d3d3d', '#3b2d23',
  '#474545', '#464645', 
  '#5f5d5c', '#535250', '#aea8a4'
]

MATERIAL_TEXTURE = [
  '#5e6276', 
  '#57586c', 
  '#9397a3', 
  '#4c4c55', 
  '#626368',
  '#5e6276', 
  '#57586c', 
  '#9397a3', 
  '#4c4c55', 
  '#626368'
]

TEXTURE_INSCRIPTIONS = [
  '#a09993', '#a3a1a1', '#a9753f', '#afa8a5',
  '#b2aca9', '#b39572', '#b47a3a',
  '#c5c3c3',
  '#d1a170', '#dfd9d6', '#d4d4ce',
  '#3e3734', '#32240a',
  '#463f3b', '#464546', '#4c3829',
  '#525151', '#544e47', '#504842', '#5e5551',
  '#6e6e69', '#625f5d', '#6f6f72', '#6b5442',
  '#726d6b', '#7f766f', '#73716d', '#796453', '#7e551e',
  '#8a8881', '#8c7c6d', '#803e06', '#8f5621',
  '#998e84', '#947c6a', '#9b897b', '#977f6a',

  '#5e6276', '#57586c', '#9397a3', '#4c4c55', '#626368',
  '#5e6276', '#57586c', '#9397a3', '#4c4c55', '#626368'
]

IMG_WIDTH  = 400
IMG_HEIGHT = 400

NR_OF_TRAINING_IMAGES    = 5000
NR_OF_VALIDATION_IMAGES  = 625
NR_OF_TESTING_IMAGES     = 625

def show_image(img):
    plt.axis('off')
    plt.imshow(img)
    plt.show()

###################
# Color functions #
###################

def prepareColors(lettersTexture, inscriptionsTexture):
    bgColors = []
    fgColors = []

    for t in lettersTexture:
        bgColors.append(ImageColor.getrgb(t))
        fgColors.append(ImageColor.getrgb(t))

    for t in inscriptionsTexture:
        bgColors.append(ImageColor.getrgb(t))

    return bgColors, fgColors

def plotRandomColorBlobs(draw, colors, count, mins, maxs):
    for i in range(count):
        x = random.randint(0, IMG_WIDTH)
        
        y = random.randint(0, IMG_HEIGHT)
        
        w = random.randint(mins, maxs)
        
        l = random.randint(mins, maxs)
        
        c = colors[random.randint(0, len(colors) - 1)]

        draw.rectangle((x, y, x + w, y + l), fill = c, outline = None)


########################
# Background functions #
########################

def getackground(path):

    files = [f for f in os.listdir(path) if f.endswith('.png')]

    background = Image.open(path + files[0])

    background = background.convert('RGBA')

    return background

# Random select of an inscription texture for background texture
def getRandomBackground(path):

    files = [f for f in os.listdir(path) if f.endswith('.png')]

    background = Image.open(path + np.random.choice(files))

    background = background.convert('RGBA')

    return background

# Random selection of a background texture
def createBackground(main_color, colors, width, height):
    image = Image.new(
        'RGBA',
        (width, height),
        main_color
    )

    drawBackground = ImageDraw.Draw(image)

    plotRandomColorBlobs(drawBackground, colors, 2800, 2, 5)

    image = image.filter(ImageFilter.MedianFilter(size = 3))

    return image


########################
# Foreground functions #
########################

def getLetterForeground(letterPath):
    foreground       = Image.open(letterPath)
    foreground_alpha = np.array(foreground.getchannel(3))

    return foreground

def getRandomLetterForeground(path):

    files = [f for f in os.listdir(path) if f.endswith('.png')]

    foreground = Image.open(path + np.random.choice(files))

    #foreground_alpha = np.array(foreground.getchannel(3))

    return foreground


# def create_letter_obstruction_foreground(colors, width, height):
#     image   = Image.new(
#         'RGBA', 
#         (width, height), 
#         (0, 0, 0, 0)
#     )
    
#     drawForeground = ImageDraw.Draw(image)
  
#     plotRandomColorBlobs(drawForeground, colors, 1, 80, 120)
  
#     image = image.filter(ImageFilter.MedianFilter(size = 3))
    
#     return image


def plotRandomLetters(color_range, count, width, height, mins, maxs):
    image = Image.new(
        'RGBA', 
        (width, height), 
        (0, 0, 0, 0)
    )
    drawLetter = ImageDraw.Draw(image)
    letterInfo = []

    for i in range(count):
        x = random.randint(0, width - 10)
        y = random.randint(0, height - 10)
        w = random.randint(mins, maxs)
        c = (random.randint(color_range[0][0], color_range[0][1]),
             random.randint(color_range[1][0], color_range[1][1]),
             random.randint(color_range[2][0], color_range[2][1]))
        
        letterInfo.append([x, y, w, w, c])

        drawLetter.ellipse((x, y, x + w, y + w), fill = c, outline = None)

    return image, letterInfo

#####################################
# Apply augmentations on the letter #
#####################################
def foregroundAugmentation(foreground):
    # Random rotation, zoom, translation
    angle = random.randint(0, 20)
    zoom  = random.random() * 0.4 + 0.8
    foreground = foreground.rotate(angle, resample = Image.BICUBIC, expand = True)
    new_size = (int(foreground.size[0] * zoom), int(foreground.size[1] * zoom))
    foreground = foreground.resize(new_size, resample=Image.BICUBIC)
    # Adjust foreground brightness
    brightness_factor = random.random() * .4 + .7
    enhancer          = ImageEnhance.Brightness(foreground)
    foreground        = enhancer.enhance(brightness_factor)
    return foreground

# Create a mask for this new foreground object
def getForegroundMask(foreground):  
    mask_new = foreground.getchannel(3)
    return mask_new

##########################
# Create layered image   #
##########################
def create_annotation(img, fruit_info, obj_name, img_name ,ann_name):
    pobjs = []

    for i in range(len(fruit_info)):
        pct    = 0
        circle = fruit_info[i]
        color  = circle[4]
        
        for i in range(circle[2]):
            if (circle[0] + i >= IMG_WIDTH):
                continue

            for j in range(circle[3]):
                if (circle[1] + j >= IMG_HEIGHT):
                    continue
                
                r = img.getpixel((circle[0] + i, circle[1] + j))
                
                if (r[0] == color[0]):
                    pct = pct + 1
        
        diffculty = pct / (circle[2] * circle[3])
        
        if (diffculty > 0.1):
            dif = True
            
            if (diffculty > 0.4):
                dif = False
            
            pobjs.append(PascalObject(
                obj_name, 
                "", 
                truncated = False, 
                difficult = dif, 
                bndbox = BndBox(
                    circle[0], 
                    circle[1],
                    circle[0] + circle[2],
                    circle[1] + circle[3]
                )
            ))

    pascal_ann = PascalVOC(
        img_name,
        size    = size_block(IMG_WIDTH, IMG_HEIGHT, 3),
        objects = pobjs
    )

    pascal_ann.save(ann_name)

##########################
# Create layered image   #
##########################

def createLayeredImage1(foreground, mask, background):
    max_xy_position = (background.size[0] - foreground.size[0], background.size[1] - foreground.size[1])
    paste_position = (random.randint(0, max_xy_position[0]), random.randint(0, max_xy_position[1]))

    # Create a new foreground image as large as the background and paste it on top
    new_foreground = Image.new('RGBA', background.size, color = (0, 0, 0, 0))
    new_foreground.paste(foreground, paste_position)

    new_alpha_mask = Image.new('L', background.size, color=0)
    new_alpha_mask.paste(mask, paste_position)
    
    composite = Image.composite(new_foreground, background, new_alpha_mask)

    # Grab the alpha pixels above a specified threshold
    alpha_threshold = 200
    mask_arr = np.array(np.greater(np.array(new_alpha_mask), alpha_threshold), dtype=np.uint8)
    hard_mask = Image.fromarray(np.uint8(mask_arr) * 255, 'L')
    
    # Get the smallest & largest non-zero values in each dimension and calculate the bounding box
    nz = np.nonzero(hard_mask)
    bbox = [np.min(nz[0]), np.min(nz[1]), np.max(nz[0]), np.max(nz[1])] 

    return composite, hard_mask, bbox


def crop_bkg(img, width, height):
    image   = Image.new(
        'RGBA', 
        (width, height), 
        (0, 0, 0, 0)
    )

    image_arr = np.asarray(img)
    
    x, y, _ =  image_arr.shape
    n1      = np.random.randint(0, 40)
    n2      = np.random.randint(n1 + 80, 120)
    n3      = np.random.randint(0, 40)
    n4      = np.random.randint(n3 + 80, 160)

    cropped = image_arr[n1:n2, n3:n4]
    cropped = cv2.resize(cropped, (75, 70))
    cropped = Image.fromarray(cropped)

    image = image.filter(ImageFilter.MedianFilter(size = 3))

    image.paste(cropped, (40, 40), cropped)

    return image

# def create_natural_image_for_testing(letterPath, letterName, i, letterTrainPath):
#     ext       = '{}_{}'.format(letterName, i)
#     imageName = '{}/texture_letter_{}.png'.format(letterTrainPath, ext)
#     maskName  = '{}/mask_letter_{}.png'.format(letterTrainPath, ext)

#     background = getRandomBackground(BACKGROUNDS_PATH)

#     foreground = getLetterForeground(letterPath)

#     newForeground = foregroundAugmentation(foreground)

#     mask_new = getForegroundMask(newForeground)

#     composite, hard_mask, bbox = createLayeredImage1(newForeground, mask_new, background)

#     composite.save(imageName)
    
#     #hard_mask.save(maskName)

#     return composite, imageName, bbox

def create_natural_image_for_testing(letterPath, letterName, i, letterTrainPath):
    ext       = '{}_{}'.format(letterName, i)
    imageName = '{}/texture_letter_{}.png'.format(letterTrainPath, ext)
    # maskName  = '{}/mask_letter_{}.png'.format(letterTrainPath, ext)

    background = getRandomBackground(BACKGROUNDS_PATH)

    foreground = getLetterForeground(letterPath)

    newForeground = foregroundAugmentation(foreground)

    mask_new = getForegroundMask(newForeground)

    try:
        composite, hard_mask, bbox = createLayeredImage1(newForeground, mask_new, background)
    except ValueError as e:
        print("Skipping image generation due to:", e)
        return None, None, None

    if composite is None:
        print("Skipping image generation due to empty composite")
        return None, None, None

    composite.save(imageName)
    # hard_mask.save(maskName)

    return composite, imageName, bbox



def create_augmented_image_for_training(letterPath, letterName, i, letterTrainPath):
    ext       = '{}_{}'.format(letterName, i)
    imageName = '{}/texture_letter_{}.png'.format(letterTrainPath, ext)
    maskName  = '{}/mask_letter_{}.png'.format(letterTrainPath, ext)

    background = getRandomBackground(BACKGROUNDS_PATH)
    
    foreground = getRandomLetterForeground(letterPath)

    obstr_foreground = crop_bkg(background, IMG_WIDTH, IMG_HEIGHT)

    aug_background = augment_bkg(background)
    aug_foreground = augment_letter(foreground)
    aug_obstr_foreground = augment_bkg(obstr_foreground)

    mask_new = getForegroundMask(aug_foreground)

    composite, hard_mask, bbox = createLayeredImage1(aug_foreground, mask_new, aug_background)

    #hard_mask.save(maskName)

    #composite.paste(aug_obstr_foreground, (40, 40), aug_obstr_foreground)

    composite = degrade_img(composite)

    return composite, imageName, bbox


def create_synthetic_image_for_training(letterPath, letterName, i, bgColors, fgColors, imagePath):
    ext       = '{}_{}'.format(letterName, i)
    imageName = '{}/synthetic_letter_{}.png'.format(imagePath, ext)
    #annName   = '{}/ann_{}.xml'.format(ann_path, ext)

    main_color = np.random.choice(TEXTURE_INSCRIPTIONS)

    background = createBackground(main_color, bgColors, IMG_WIDTH, IMG_HEIGHT)

    foreground = getRandomLetterForeground(letterPath)

    obstr_foreground = crop_bkg(background, IMG_WIDTH, IMG_HEIGHT)

    aug_background = augment_bkg(background)
    aug_foreground = augment_letter(foreground)
    obstr_foreground = augment_bkg(obstr_foreground)

    mask_new = getForegroundMask(aug_foreground)

    composite, hard_mask, bbox = createLayeredImage1(aug_foreground, mask_new, aug_background)

    #composite.paste(obstr_foreground, (0, 0), obstr_foreground)

    composite = degrade_img(composite)

    #hard_mask.save(maskName)

    #create the anootation File
    #create_annotation(image, lettersInfo, 'oranges', imageName, ann_name)

    return composite, imageName, bbox



###########################
# Create Training Dataset #
###########################

def generate_training_dataset(nrOfImages):
    bgColors, fgColors = prepareColors(TEXTURE_LETTERS, TEXTURE_INSCRIPTIONS)
    lettersFolder   = [f for f in os.listdir(LETTERS_VARIATION_PATH) if os.path.isdir(os.path.join(LETTERS_VARIATION_PATH, f))]
    csv_lines          = []

    for f in lettersFolder:
        letterName = f

        print(letterName)

        os.mkdir(DATASET_PATH + 'train/' + letterName)

        print('Generate images for letter: ' + letterName)

        for i in range(nrOfImages):
            syImage, syImageName, bbox = create_synthetic_image_for_training(
                LETTERS_VARIATION_PATH + letterName + "/",
                letterName,
                i, 
                bgColors,
                fgColors,
                DATASET_PATH + 'train/' + letterName
            )

            syImage.save(syImageName)
            csv_lines.append([syImageName, letterName, bbox[0], bbox[1], bbox[2], bbox[3]]) #mask_path

            txImage, txImageName, bbox = create_augmented_image_for_training(
                LETTERS_VARIATION_PATH + letterName + "/",
                letterName,
                i,
                DATASET_PATH + 'train/' + letterName
            )

            txImage.save(txImageName)
            csv_lines.append([txImageName, letterName, bbox[0], bbox[1], bbox[2], bbox[3]]) #mask_path

    with open(train_annotations_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        
        for csv_line in csv_lines:
            writer.writerow(csv_line)

#############################
# Create Validation Dataset #
#############################

def generate_validation_dataset(nrOfImages):
    bgColors, fgColors = prepareColors(TEXTURE_LETTERS, TEXTURE_INSCRIPTIONS)
    letterNames        = [f for f in os.listdir(LETTERS_VARIATION_PATH) if os.path.isdir(os.path.join(LETTERS_VARIATION_PATH, f))]
    csv_lines          = []

    for letter_name in letterNames:
        print(letter_name)
        
        os.mkdir(DATASET_PATH + 'valid/' + letter_name)

        print('Generate validation images for the letter: ' + letter_name)

        for i in range(nrOfImages):
            syImage, syImageName, bbox = create_synthetic_image_for_training(
                LETTERS_VARIATION_PATH + letter_name + "/",
                letter_name,
                i, 
                bgColors,
                fgColors,
                DATASET_PATH + 'valid/' + letter_name
            )

            syImage.save(syImageName)
            csv_lines.append([syImageName, letter_name, bbox[0], bbox[1], bbox[2], bbox[3]]) #mask_path

            txImage, txImageName, bbox = create_augmented_image_for_training(
                LETTERS_VARIATION_PATH + letter_name + "/",
                letter_name,
                i,
                DATASET_PATH + 'valid/' + letter_name
            )

            txImage.save(txImageName)
            csv_lines.append([txImageName, letter_name, bbox[0], bbox[1], bbox[2], bbox[3]]) #mask_path

    with open(valid_annotations_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        
        for csv_line in csv_lines:
            writer.writerow(csv_line)

###########################
# Create Testing Dataset  #
###########################

def generate_testing_dataset(nrOfImages):
    bgColors, fgColors = prepareColors(TEXTURE_LETTERS, TEXTURE_INSCRIPTIONS)
    letterNames        = [f for f in os.listdir(LETTERS_VARIATION_PATH) if os.path.isdir(os.path.join(LETTERS_VARIATION_PATH, f))]
    csv_lines          = []

    for letter_name in letterNames:
        print(letter_name)
        
        os.mkdir(DATASET_PATH + 'test/' + letter_name)

        print('Generate testing images for the letter: ' + letter_name)

        for i in range(nrOfImages):
            syImage, syImageName, bbox = create_synthetic_image_for_training(
                LETTERS_VARIATION_PATH + letter_name + "/",
                letter_name,
                i, 
                bgColors,
                fgColors,
                DATASET_PATH + 'test/' + letter_name
            )

            syImage.save(syImageName)
            csv_lines.append([syImageName, letter_name, bbox[0], bbox[1], bbox[2], bbox[3]]) #mask_path

            txImage, txImageName, bbox = create_augmented_image_for_training(
                LETTERS_VARIATION_PATH + letter_name + "/",
                letter_name,
                i,
                DATASET_PATH + 'test/' + letter_name
            )

            txImage.save(txImageName)
            csv_lines.append([txImageName, letter_name, bbox[0], bbox[1], bbox[2], bbox[3]]) #mask_path

    with open(valid_annotations_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        
        for csv_line in csv_lines:
            writer.writerow(csv_line)
    # lettersFileNames = [f for f in os.listdir(LETTERS_PATH) if f.endswith('.png')]

    # for f in lettersFileNames:
    #     pathname, extension = os.path.splitext(f)

    #     letterName = pathname.split('/')[-1]

    #     os.mkdir(DATASET_PATH + 'test/' + letterName)

    #     print('Generate images for letter: ' + letterName)

    #     for i in range(nrOfImages):
    #         txImage, txImageName, bbox = create_natural_image_for_testing(
    #             LETTERS_PATH + f,
    #             letterName,
    #             i,
    #             DATASET_PATH + 'test/' + letterName
    #         )

def setup_folder_structure() -> None:
    # Create base folders if they don't exist

    if not dir_data.exists():  dir_data.mkdir()
    if not dir_train.exists(): dir_train.mkdir()
    if not dir_valid.exists(): dir_valid.mkdir()
    if not dir_test.exists():  dir_test.mkdir()

    #Print the directory structure
    dir_str = os.system('''ls -R data | grep ":$" | sed -e 's/:$//' -e 's/[^-][^\/]*\//--/g' -e 's/^/   /' -e 's/-/|/' ''')
    print(dir_str)

    return

def main():
    setup_folder_structure()

    generate_training_dataset(NR_OF_TRAINING_IMAGES)

    generate_validation_dataset(NR_OF_VALIDATION_IMAGES)

    generate_testing_dataset(NR_OF_TESTING_IMAGES)

if __name__ == "__main__":
    main()