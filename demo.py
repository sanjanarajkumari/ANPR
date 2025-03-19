import cv2
from matplotlib import pyplot as plt
import numpy as np
import easyocr
import imutils

# Read the image
img = cv2.imread("Test3.jpg")
if img is None:
    print("Error: Could not read the image.")
    exit()

# Display the original image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
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
edged = cv2.Canny(bfilter, 50, 250)  # Adjusted edge detection parameters

# Display the edge-detected image
plt.imshow(edged, cmap='gray')
plt.title("Edge Detection")
plt.axis('off')
plt.show()

# Find contours
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Take top 10 contours

# Debugging: Print total contours found
print(f"Total contours found: {len(contours)}")

# Loop over contours to find a rectangular one
location = None
for i, contour in enumerate(contours):
    approx = cv2.approxPolyDP(contour, 10, True)
    print(f"Contour {i}: {len(approx)} points")  # Debugging

    if len(approx) == 4:
        location = approx
        print("Number plate detected!")
        break

if location is None:
    print("Error: No valid contour found for the number plate.")
    exit()

# Convert location to integer type
location = location.astype(int)

# Draw the detected plate area
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

# Display the image with detected number plate
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title("Detected Plate Area")
plt.axis('off')
plt.show()

# Crop the detected number plate
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

# Display the cropped number plate
plt.imshow(cropped_image, cmap='gray')
plt.title("Cropped Number Plate")
plt.axis('off')
plt.show()

# OCR to extract text from the number plate
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)

# Display OCR result
if result:
    text = result[0][-2]  # Extract text from result
    print("Recognized Text:", text)
else:
    print("Error: No text detected.")
    exit()

# Overlay the recognized text on the original image
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(location[0][0][0], location[1][0][1] + 60),
                  fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

# Display the final image with recognized text
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.title("Final Result with OCR Text")
plt.axis('off')
plt.show()
