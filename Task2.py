import cv2
import pytesseract
import numpy as np

def get_average_color(image, x, y, w, h):
    """Calculate the average color of a specific region in the image."""
    region = image[y:y+h, x:x+w]
    average_color_per_row = np.mean(region, axis=0)
    average_color = np.mean(average_color_per_row, axis=0)
    return (int(average_color[0]), int(average_color[1]), int(average_color[2]))

def annotate_image(image_path):
    """Annotate the image with each cell's details."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at path {image_path}")
        return None

    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out too small rectangles which might be noise
        if w > 50 and h > 20:  # Adjust size thresholds as needed
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = image[y:y+h, x:x+w]
            text = pytesseract.image_to_string(roi, config='--psm 6').strip()
            avg_color = get_average_color(image, x, y, w, h)

            # Extract text size (based on height of bounding box)
            text_size = h / len(text.split('\n')) if text else 0  # Simple estimation of text size

            # Annotations
            info_text = f"Size: {w}x{h}px, Color: {avg_color}, Text size: {text_size:.2f}px"
            cv2.putText(output, info_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(output, text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    return output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ensure this is correct

# Process the image and display the result
processed_image = annotate_image('Excel.jpeg')  # Update this with your image path
if processed_image is not None:
    cv2.imshow('Annotated Image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to process the image.")
