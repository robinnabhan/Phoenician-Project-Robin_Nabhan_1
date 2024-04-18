
import cv2
import numpy as np

# Load the image
image = cv2.imread(r'C:\Users\sim-robinnab\Desktop\Phoenician-Project-Robin_Nabhan\Test\Phoenician_inscription_alanya.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply contrast stretching
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Remove noise and smooth the image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

# Save the preprocessed image
cv2.imwrite('preprocessed_inscription.jpg', enhanced)
# Display the original and preprocessed images
cv2.imshow('Original Image', image)
cv2.imshow('Preprocessed Image', enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()