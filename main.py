import image_processing
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import prediction_model

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


# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
# https://www.cs.tau.ac.il/~wolf/papers/CNNNGram.pdf

# For line separation
# https://ciir.cs.umass.edu/pubfiles/mm-27.pdf current guide/paper

# For word separation
# https://sci-hub.se/10.1109/ICDAR.1995.598958

# both?
# https://maroo.cs.umass.edu/getpdf.php?id=505

# https://pd.daffodilvarsity.edu.bd/course/material/book-430/pdf_content convex hull

# https://sci-hub.se/https://ieeexplore.ieee.org/document/8902434

# to increase handwriting samples can apply random rotations, crops, and noise/deterioration

# want to apply the original word detection (segment_words) to the separated lines, also potentially want to increase
# the line height by a few pixels so that ascenders and descenders are not missed

# might want to extract words from greyscale images so that it still has the background, and it doesn't look off,
# due to the contrast, could do this by storing the location of the words in the line and then using the position of
# the line relative to the whole image to then extract the image of the word straight from the original greyscale image

# could modify an existing dataset and include samples of my handwriting in it OR make my own dataset just for my
# handwriting, trade off of size of datasets vs accuracy on just my handwriting
# word list for my database https://www.ef.co.nz/english-resources/english-vocabulary/top-1000-words/

def main():
    # image_processing.write_words_from_image('uni_sample_handwriting', 'words_to_label')
    pass


if __name__ == "__main__":

    items = os.listdir("labelled_images_school")
    items1 = os.listdir("labelled_images_geog")
    items2 = os.listdir("labelled_images")
    print("Labelled:")
    # print(len(items))
    print(len(items)+len(items1)+len(items2))
    items3 = os.listdir("words_to_label")
    print("To Label:")
    print(len(items3))
    # started 18th with 9399 to label
