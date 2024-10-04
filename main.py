import cv2
import numpy as np
from sklearn.cluster import KMeans

#import billeder
img = cv2.imread("1.jpg")
imgHvid = img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.blur(img, (6, 6))
Grayimg = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)

Crown_N = cv2.imread("Crown_N.png")
Crown_S = cv2.imread("Crown_S.png")
Crown_E = cv2.imread("Crown_E.png")
Crown_W = cv2.imread("Crown_W.png")

template_height, template_width = Crown_N.shape[:2]
template_height1, template_width1 = Crown_S.shape[:2]
template_height2, template_width2 = Crown_E.shape[:2]
template_height3, template_width3 = Crown_W.shape[:2]

result = cv2.matchTemplate(gray_blurred, Crown_N, cv2.TM_CCOEFF_NORMED)
result1 = cv2.matchTemplate(gray_blurred, Crown_S, cv2.TM_CCOEFF_NORMED)
result2 = cv2.matchTemplate(gray_blurred, Crown_E, cv2.TM_CCOEFF_NORMED)
result3 = cv2.matchTemplate(gray_blurred, Crown_W, cv2.TM_CCOEFF_NORMED)

threshold_N = 0.586# Adjust this value based on your needs
yloc_N, xloc_N = np.where(result >= threshold_N)

threshold_S = 0.572
yloc_S, xloc_S = np.where(result1 >= threshold_S)

threshold_E = 0.72
yloc_E, xloc_E = np.where(result2 >= threshold_E)

threshold_W = 0.642
yloc_W, xloc_W = np.where(result3 >= threshold_W)

# Assuming xloc_N and yloc_N are defined and img, template_width, template_height are initialized
xPos = []
yPos = []
count = 0

for (x, y) in zip(xloc_N, yloc_N):
    xPos.append(x)
    yPos.append(y)
    cv2.circle(gray_blurred, (x + template_width // 2, y + template_height // 2), 5, (255, 255, 255), -1)
    #print(x, y)


for (x, y) in zip(xloc_S, yloc_S):
    cv2.circle(gray_blurred, (x + template_width1 // 2, y + template_height1 // 2), 5, (255, 255, 255), -1)


for (x, y) in zip(xloc_E, yloc_E):
    cv2.circle(gray_blurred, (x + template_width2 // 2, y + template_height2 // 2), 5, (255, 255, 255), -1)


for (x, y) in zip(xloc_W, yloc_W):
    cv2.circle(gray_blurred, (x + template_width3 // 2, y + template_height3 // 2), 5, (255, 255, 255), -1)

height = 500
width = 500
box_size = 100
all_colors = []

dominant_colors = {}

# Loop through the image in steps of box_size
for y in range(0, height, box_size):
    for x in range(0, width, box_size):
        # Extract the ROI
        roi = gray_blurred[y:y + box_size, x:x + box_size]

        # Reshape the ROI to be a list of pixels
        pixels = roi.reshape(-1, 3)

        # Use KMeans to find the dominant color
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(pixels)

        # Get the dominant color
        dominant_color = kmeans.cluster_centers_[0].astype(int)

        # Store the dominant color in the dictionary with box coordinates
        dominant_colors[(x, y)] = dominant_color

# Print the dominant colors for each box
for (x, y), color in dominant_colors.items():
    #print(f'Dominant color for box at ({x}, {y}) (BGR): {color}')
    # Convert to RGB format for display if needed
    print(f'Dominant color (RGB): {color[::-1]}')
cv2.imshow('Detected Hearts', gray_blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()