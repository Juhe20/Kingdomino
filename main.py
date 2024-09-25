import cv2
import numpy as np
from debugpy.server.cli import in_range

#import billede af brættet
img = cv2.imread("1.jpg")

Grayimg = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)


#import billde af krone vendt på alle måder
Crown_N = cv2.imread("Crown_N.png", cv2.IMREAD_GRAYSCALE)

#Make tiles size
M = img.shape[0]//5
N = img.shape[1]//5


tiles = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]



property = []


template_height, template_width = Crown_N.shape[:2]

#
#average = img.mean(axis=0).mean(axis=0)

# 5x5 grid
for i in range(5):
    for j in range(5):
        pass

#
result = cv2.matchTemplate(Grayimg, Crown_N, cv2.TM_CCOEFF_NORMED)

#Add all property scores
for score in property:
    score += score




threshold = 0.62  # Adjust this value based on your needs
yloc, xloc = np.where(result >= threshold)

# Draw circles on detected positions
for (x, y) in zip(xloc, yloc):
    cv2.circle(Grayimg, (x + template_width // 2, y + template_height // 2), 5, (255, 255, 255), -1)

cv2.imshow('Correlation Image', result)
cv2.imshow('Detected Hearts', Grayimg)

cv2.waitKey(0)
cv2.destroyAllWindows()


