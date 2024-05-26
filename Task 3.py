import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

    # Save annotated image
    annotated_image_path = 'annotated_image_with_ocr.png'
    cv2.imwrite(annotated_image_path, output)

    # Convert the image to PIL for final overlay
    pil_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()

    # Annotate image with OCR details
    ocr_result = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    for i in range(len(ocr_result['text'])):
        if ocr_result['text'][i].strip() != '':
            x, y, w, h = ocr_result['left'][i], ocr_result['top'][i], ocr_result['width'][i], ocr_result['height'][i]
            text = ocr_result['text'][i]
            overlay_text = f"{text} ({w}x{h})"
            draw.text((x, y + h + 5), overlay_text, fill="red", font=font)  # Adjusted position to below the box

    # Save the final image with overlays
    final_image_path = 'final_annotated_image_with_ocr.png'
    pil_image.save(final_image_path)
    pil_image.show()

    return final_image_path

# Ensure pytesseract is correctly configured
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ensure this is correct

# Process the image and display the result
annotate_image('sheet2.jpg')  # Update this with your image path
