import image_processing


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

image_processing.process_images_in_folder('sample_handwriting','contrast_handwriting')
