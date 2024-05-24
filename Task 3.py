import cv2
import pytesseract
import numpy as np

def get_average_color(image, x, y, w, h):
    region = image[y:y+h, x:x+w]
    average_color_per_row = np.mean(region, axis=0)
    average_color = np.mean(average_color_per_row, axis=0)
    average_color = tuple([int(x) for x in average_color])
    return average_color

# Configure pytesseract to use the executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

# Load the image
image = cv2.imread('sheet2.jpg')  # Update this path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detecting rectangles using contour detection
ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    x, y, w, h = cv2.boundingRect(approx)

    # Filter for rectangular areas
    if len(approx) == 4:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi = gray[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi)
        average_color = get_average_color(image, x, y, w, h)
        
        # Estimate text size based on the height of the bounding box
        text_size_estimate = h / len(text.split('\n')) if text.strip() else 0

        # Prepare text for average color, dimensions, and estimated text size
        color_text = f'Color: {average_color}'
        dim_text = f'Dim: {w}x{h}'
        size_text = f'Size (est.): {text_size_estimate:.2f} px'
        text_details = f'Text: {text.strip()}'

        # Positioning text on image
        cv2.putText(image, color_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(image, dim_text, (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(image, size_text, (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(image, text_details, (x, y + h + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Display the result
cv2.imshow('Rectangles Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
