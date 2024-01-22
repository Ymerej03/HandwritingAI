import matplotlib.pyplot as plt
import image_processing
import data_enlargment
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
# The goal of the project is to be able to take images as input (in my handwriting) and for a machine learning model
# to be able to "read" what is written and convert this to a text file.
# The model (maybe separate) should also be able to take a text file as input and produce an image of handwritten words
# in my handwriting.

# idea for the workflow. folder to store the raw images of sample handwriting. these are black and whited and put into
# a new folder. then the images are split from paragraphs to individual words. these snippets are then put in their own
# folder. then there is a UI/captcha style system where each word is plotted on the screen and there is a text box for
# the user to type the word they see in. each of these words is also appended with a strictly increasing number. e.g.
# word0, tiny1, word2, plane3 etc. these words are now fed into the AI to do AI things (the training of the model)
# a model will be saved so that the process doesn't have to happen each time you want to run (customizable)
# then maybe a user interface where you can input text, and it outputs the handwriting, or you can choose to input an
# image of handwriting, and it outputs the text.

# I think I need to edit the process images in folder function so that it doesn't use the remove hor lines function
# and then write all my samples on plain paper.

# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_tutorials.html

# https://www.cs.tau.ac.il/~wolf/papers/CNNNGram.pdf

# https://ciir.cs.umass.edu/pubfiles/mm-27.pdf current guide/paper

# to increase handwriting samples can apply random rotations, crops, and noise/deterioration

# want to apply the original word detection (segment_words) to the separated lines, also potentially want to increase
# the line height by a few pixels so that ascenders and descenders are not missed

def main():
    image_processing.process_images_in_folder('sample_handwriting', 'contrast_handwriting', True)

    # image_processing.segment_words('contrast_handwriting/processed0_numbers.jpeg', 'individual_numbers', 200, 5000)

    # image = cv2.imread('sample_handwriting/image1.jpeg')
    # image_test = image_processing.hor_line_removal(image)
    # plt.imshow(image_test)
    # plt.show()
    # data_enlargment.image_rotator('individual_letters/B')
    # image = cv2.imread('contrast_handwriting/processed1_image1.jpeg')
    # split_image = image_processing.line_splitter(image)
    # plt.imshow(split_image, cmap='gray')
    # plt.show()


if __name__ == "__main__":
    main()

