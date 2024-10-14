import cv2
import numpy as np
from fontTools.ttx import waitForKeyPress
from plotly.express import imshow
from sklearn.cluster import KMeans
tot = 0
RealScores = [
    36,
    43,
    52,
    42,
    36,
    43,
    52,
    42,
    45,
    34,
    49,
    22,
    45,
    34,
    49,
    22,
    20,
    60,
    36,
    52,
    20,
    60,
    36,
    52,
    44,
    48,
    67,
    59,
    44,
    48,
    67,
    59,
    21,
    36,
    46,
    51,
    21,
    36,
    46,
    51,
    33,
    43,
    66,
    33,
    38,
    43,
    66,
    45,
    26,
    34,
    37,
    45,
    23,
    34,
    37,
    44,
    64,
    36,
    37,
    44,
    64,
    36,
    37,
    66,
    80,
    124,
    99,
    66,
    124,
    99,
    66,
    80,
    124,
    99]
q = 0

for q in range(74):
    # Import images
    img = cv2.imread(f"Pictures/{q+1}.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(img, (5, 5))

    Crown_N = cv2.imread("Pictures/Crown_N.png")
    Crown_S = cv2.imread("Pictures/Crown_S.png")
    Crown_E = cv2.imread("Pictures/Crown_E.png")
    Crown_W = cv2.imread("Pictures/Crown_W.png")

    # List of crown templates
    crown_templates = [Crown_N, Crown_S, Crown_E, Crown_W]
    thresholds = [0.612, 0.62, 0.65, 0.65]  # Adjust these thresholds for each template

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


    ### Finding the dominant color of each tile
    height = 500
    width = 500
    box_size = 100
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
            roi = gray_blurred[y:y + box_size, x:x + box_size]
            pixels = roi.reshape(-1, 3)
            kmeans = KMeans(n_clusters=1)
            kmeans.fit(pixels)
            dominant_color = kmeans.cluster_centers_[0].astype(int)
            dominant_colors[(x, y)] = dominant_color


            if all([0, 50, 0][i] <= dominant_color[::-1][i] <= [70, 50, 70][i] for i in range(len(dominant_color))):
                # Mine None
                Mine.append((x, y))
                cv2.rectangle(gray_blurred, (x, y), (x + 25, y + 25), (255, 255, 0), 2)
            elif all([103, 120, 0][i] <= dominant_color[::-1][i] <= [199, 188, 30][i] for i in range(len(dominant_color))):
                # Wheat Just right
                Wheat.append((x, y))
                cv2.rectangle(gray_blurred, (x, y), (x + 25, y + 25), (0, 255, 255), 2)
            elif all([0, 50, 80][i] <= dominant_color[::-1][i] <= [85, 110, 255][i] for i in range(len(dominant_color))):
                # Sea Just right
                Sea.append((x, y))
                cv2.rectangle(gray_blurred, (x, y), (x + 25, y + 25), (255, 0, 255), 2)
            elif all([12, 10, 10][i] <= dominant_color[::-1][i] <= [81, 77, 43][i] for i in range(len(dominant_color))):
                # Forest too Good
                Forest.append((x, y))
                cv2.rectangle(gray_blurred, (x, y), (x + 25, y + 25), (255, 255, 255), 2)
            elif all([70, 100, 0][i] <= dominant_color[::-1][i] <= [190, 187, 64][i] for i in range(len(dominant_color))):
                # Plains
                Plains.append((x, y))
                cv2.rectangle(gray_blurred, (x, y), (x + 25, y + 25), (0, 0, 0), 2)
            elif all([60, 70, 30][i] <= dominant_color[::-1][i] <= [130, 125, 80][i] for i in range(len(dominant_color))):
                # Swamp
                Swamp.append((x, y))
                cv2.rectangle(gray_blurred, (x, y), (x + 25, y + 25), (255, 0, 0), 2)
            else:
                Base = Base + 1
                cv2.rectangle(gray_blurred, (x, y), (x + 25, y + 25), (100, 100, 100), 2)


    # Function to check if two points are adjacent
    def are_adjacent(p1, p2):
        return (abs(p1[0] - p2[0]) == 100 and p1[1] == p2[1]) or (abs(p1[1] - p2[1]) == 100 and p1[0] == p2[0])

    # Updated function: returns connected tiles and adds them to a group
    def count_connected_tiles(start_point, visited, tile_points, group):
        stack = [start_point]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                group.append(current)
                for point in tile_points:
                    if point not in visited and are_adjacent(current, point):
                        stack.append(point)

    # Modified function to return groups of connected tiles
    def find_connected_components(tile_points):
        visited = set()
        connected_groups = []
        for point in tile_points:
            if point not in visited:
                group = []
                count_connected_tiles(point, visited, tile_points, group)
                connected_groups.append(group)
        return connected_groups

    # Updated function: returns the number of crowns on a specific tile
    def count_crowns_on_tile(tile_position, crown_positions):
        crowns_on_tile = 0
        for crown in crown_positions:
            if crown[0] >= tile_position[0] and crown[0] < tile_position[0] + box_size and \
               crown[1] >= tile_position[1] and crown[1] < tile_position[1] + box_size:
                crowns_on_tile += 1
        return crowns_on_tile

    # Updated function: calculates the score considering multiple crowns on the same tile
    def calculate_score(connected_tiles, crown_positions):
        score = 0
        for group in connected_tiles:
            total_crowns_in_group = sum(count_crowns_on_tile(tile, crown_positions) for tile in group)
            if total_crowns_in_group > 0:
                score += total_crowns_in_group * len(group)  # Crowns count multiplied by the number of connected tiles
            # No points are added if there are no crowns in the group
        return score

    # Example usage for multiple tile types
    tile_types = {
        "Forest": Forest,
        "Wheat": Wheat,
        "Sea": Sea,
        "Mine": Mine,
        "Plains": Plains,
        "Swamp": Swamp,
    }

    # Total score calculation
    total_score = 0
    for tile_name, tile_points in tile_types.items():
        connected_groups = find_connected_components(tile_points)
        score = calculate_score(connected_groups, unique_crowns)
        total_score += score
        #print(f"Score for {tile_name}: {score}")
    Realscore = RealScores[q]
    print(f"The IMGP total score for board{q}: {total_score}")
    print(f"The counted total score for board{q}: {Realscore}")
    t = total_score / Realscore
    print(f"The difference is: {t:.2f}")
    q = q + 1
    tot +=t
    errors = Realscore - total_score

    # Step 2: Calculate the mean error
    mean_error = np.mean(errors)

mean_error/len(RealScores)
print("Mean Error:", mean_error)

print(tot)
