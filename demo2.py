import cv2
import pytesseract
from matplotlib import pyplot as plt
import numpy as np
import imutils

# Set Tesseract-OCR path (Only for Windows users)
# Change the path if Tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Read the image
img = cv2.imread("Test3.jpg")

# Display the original image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter for noise reduction
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

# Display the processed grayscale image
plt.imshow(bfilter, cmap='gray')
plt.title('Processed Image')
plt.axis('off')
plt.show()

# Apply Canny Edge Detection
edged = cv2.Canny(bfilter, 30, 200)

# Display the edge-detected image
plt.imshow(edged, cmap='gray')
plt.title("Edge Detection")
plt.axis('off')
plt.show()

# Find contours
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Find the best contour
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

if location is None:
    print("Error: No valid contour found for the number plate.")
    exit()

print("Location:", location)

# Create a mask and extract the number plate
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

# Display the detected plate area
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title("Detected Plate Area")
plt.show()

# Crop the number plate
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

# Resize for better OCR accuracy
cropped_image = cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Display the cropped image
plt.imshow(cropped_image, cmap='gray')
plt.title("Cropped Number Plate")
plt.show()

# OCR using Tesseract
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
text = pytesseract.image_to_string(cropped_image, config=custom_config).strip()

if text:
    print("Recognized Text:", text)
else:
    print("Error: No text detected.")
    exit()

# Post-processing for OCR correction
corrections = {
   # "H": "M",  # Fixes H being mistaken for M
    "0": "O",  # Fixes 0 being mistaken for O
    "1": "I",  # Fixes 1 being mistaken for I
}
for wrong, correct in corrections.items():
    text = text.replace(wrong, correct)

print("Corrected Text:", text)

# Draw detected text on image
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(location[0][0][0], location[1][0][1] + 60),
                   fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

# Show final result with detected text
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.title("Final Result with Corrected OCR")
plt.show()
