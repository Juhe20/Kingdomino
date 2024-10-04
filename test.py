import cv2
import numpy as np
from astropy.io.fits import append
from sklearn.cluster import KMeans

#import billeder
img = cv2.imread("1.jpg")
imgHvid = img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.blur(img, (5, 5))
Grayimg = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)

Crown_N = cv2.imread("Crown_N.png")
Crown_S = cv2.imread("Crown_S.png")
Crown_E = cv2.imread("Crown_E.png")
Crown_W = cv2.imread("Crown_W.png")

# List of crown templates
crown_templates = [Crown_N, Crown_S, Crown_E, Crown_W]
thresholds = [0.62, 0.62, 0.65, 0.65]  # Adjust these thresholds for each template

# Function to detect crowns using template matching
def detect_crowns(img, templates, thresholds):
    detected_positions = []
    for template, threshold in zip(templates, thresholds):
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        yloc, xloc = np.where(result >= threshold)
        for (x, y) in zip(xloc, yloc):
            detected_positions.append((x, y))
    return detected_positions

# Function to filter out duplicate detections
def filter_duplicates(detected_positions, min_distance=25):
    unique_positions = []
    for (x, y) in detected_positions:
        if all(np.sqrt((x - ux) ** 2 + (y - uy) ** 2) >= min_distance for (ux, uy) in unique_positions):
            unique_positions.append((x, y))
    return unique_positions

# Detect crowns
detected_crowns = detect_crowns(gray_blurred, crown_templates, thresholds)

# Filter out duplicate detections
unique_crowns = filter_duplicates(detected_crowns)

CrownPositions = []
# Function to draw detected crowns on the image
def draw_detected_crowns(img, detected_crowns):
    for (x, y) in detected_crowns:
        CrownPositions.append((x, y))
        # Draw a rectangle around the detected crown
        cv2.rectangle(img, (x, y), (x + 25, y + 25), (0, 255, 0), 2)  # Green rectangle
        # Optionally, overlay a crown template
        # img[y:y + 25, x:x + 25] = Crown_N  # Use appropriate crown template
###Finding the dorminant color of each tile
height = 500
width = 500
box_size = 100
all_colors = []

dominant_colors = {}
Forest = []
Wheat = []
Sea = []
Mine = []
Plains = []
Swamp = []
Base = 0

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
    #print(f'Dominant color (RGB): {color[::-1]} at {(x, y)}')

    # Check each tile color
    if all([41,54,15][i] <= color[::-1][i] <= [67, 73, 38][i] for i in range(len(color))):
        # Forest
        Forest.append((x, y))
        cv2.rectangle(gray_blurred, (x, y), (x + 25, y + 25), (255, 255, 255), 2)
    elif all([158,130,0][i] <= color[::-1][i] <= [199,168, 17][i] for i in range(len(color))):
        #Wheat
        Wheat.append((x, y))
        cv2.rectangle(gray_blurred, (x, y), (x + 25, y + 25), (0, 255, 255), 2)
    elif all([0, 70, 110][i] <= color[::-1][i] <= [80, 100, 200][i] for i in range(len(color))):
        # Sea
        Sea.append((x, y))
        cv2.rectangle(gray_blurred, (x, y), (x + 25, y + 25), (255, 0, 255), 2)
    elif all([50,40,10][i] <= color[::-1][i] <= [75,75, 32][i] for i in range(len(color))):
        #Mine
        Mine.append((x, y))
        cv2.rectangle(gray_blurred, (x, y), (x + 25, y + 25), (255, 255, 0), 2)
    elif all([80,110,10][i] <= color[::-1][i] <= [150,182, 60][i] for i in range(len(color))):
        #Plains
        Plains.append((x, y))
        cv2.rectangle(gray_blurred, (x, y), (x + 25, y + 25), (0, 0, 0), 2)
    elif all([60,70,30][i] <= color[::-1][i] <= [120,125, 80][i] for i in range(len(color))):
        #Swamp
        Swamp.append((x, y))
        cv2.rectangle(gray_blurred, (x, y), (x + 25, y + 25), (255, 0, 0), 2)
    else:
        Base = Base+1
        cv2.rectangle(gray_blurred, (x, y), (x + 25, y + 25), (100, 100, 100), 2)


#print(Forest) #White
#print(Swamp) #Blue
#print(Plains) #Black
#print(Mine) #Cyan
#print(Sea) #Pink
#print(Base) #Gray

#Are they next to eachother?
# Function to check if two points are adjacent
def are_adjacent(p1, p2):
    return (abs(p1[0] - p2[0]) == 100 and p1[1] == p2[1]) or (abs(p1[1] - p2[1]) == 100 and p1[0] == p2[0])

def count_connected_tiles(start_point, visited, tile_points):
    stack = [start_point]
    count = 0

    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            count += 1
            # Check adjacent points for the specific tile type
            for point in tile_points:
                if point not in visited and are_adjacent(current, point):
                    stack.append(point)
    return count

def find_connected_components(tile_points):
    visited = set()
    connected_counts = []

    for point in tile_points:
        if point not in visited:
            count = count_connected_tiles(point, visited, tile_points)
            connected_counts.append(count)

    return connected_counts

# Example usage for multiple tile types
tile_types = {
    "Forest": Forest,
    "Wheat": Wheat,
    "Sea": Sea,
    "Mine": Mine,
    "Plains": Plains,
    "Swamp": Swamp,
}

# Get the counts of connected tiles for each type
for tile_name, tile_points in tile_types.items():
    connected_counts = find_connected_components(tile_points)
    print(f"Number of connected tile groups for {tile_name}:")
    for count in connected_counts:
        print(count)

draw_detected_crowns(gray_blurred, unique_crowns)
cv2.imshow('Detected Hearts', gray_blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()