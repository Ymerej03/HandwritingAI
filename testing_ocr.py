import pytesseract
from pytesseract import Output
import cv2

# specifying the executable location
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# # reads the image and draws bounding boxes around the located words
# img = cv2.imread('contrast_handwriting/processed0_image3.jpeg')
# d = pytesseract.image_to_data(img, output_type=Output.DICT)
# n_boxes = len(d['level'])
# for i in range(n_boxes):
#     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# cv2.imwrite('test_output_image3.png', img)

print(pytesseract.image_to_string('contrast_handwriting/processed0_image3.jpeg'))
