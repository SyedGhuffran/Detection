import cv2
import numpy as np
import webcolors

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(requested_color):
    try:
        closest_name = webcolors.rgb_to_name(requested_color)
    except ValueError:
        closest_name = closest_color(requested_color)
    return closest_name

# Read the input image
image = cv2.imread('Circle.jpeg', cv2.IMREAD_COLOR)

if image is None:
    print("Image not found or unable to open")
else:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            
            # Get the color of the circle's center
            color = image[center[1], center[0]].tolist()  # [B, G, R]
            color_name = get_color_name((color[2], color[1], color[0]))  # Convert to (R, G, B) and get color name
            
            # Draw the circle on the original image
            cv2.circle(image, center, radius, (0, 255, 0), 2)
            
            # Draw the text displaying size and color with unit
            text = f"Size: {radius} pixels, Color: {color_name}"
            cv2.putText(image, text, (center[0] - 50, center[1] - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Display circle's size and color in console
            print(f"Circle detected - Size: {radius} pixels, Color: {color_name}")

    # Display the image with detected circles
    cv2.imshow('Detected Circles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
