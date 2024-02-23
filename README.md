# HandwritingAI

HandwritingAI is a Python project to transcribe words from images of my handwriting. 

The goal of this project is to be able to take images as input (in my handwriting) and for a machine learning model to be able to read what is written and convert this to text.

Images are read in where they are processed to be black and white, these images are then split into lines and then the lines are further split into images of individual words. These individual words can be saved to a folder to be labeled for tailored dataset creation for model training, or they can be passed to a pre-trained model to be transcribed.

This project includes some pre-trained models most of which are trained on my handwriting specifically however there is also a model trained on the [IAM words dataset](https://doi.org/10.1007/s100320200071). You can use these models to transcribe images, or you can create your own dataset which is semi-automated. It should be noted that I started seeing "passable" transcriptions at a sample size of approximately 5000 words.
## Installation
Clone this repository and then depending on what packages you already have installed you may need to install the following using the package manager [pip](https://pip.pypa.io/en/stable/) or other methods.

```bash
pip install numpy
pip install opencv-python
pip install scipy
pip install mltu
pip install tk
pip install pillow
```

## Usage
If you just want to transcribe an image using either my specific handwriting model or the more general IAM trained model then you should just run main.py and follow the instructions that pop up in the terminal.

However, if you are wanting to train your own model then you will need lots of images containing your writing. You will need to run write_words_from_image() (this is quite slow takes roughly 25 seconds per image) and then run dataset_creation.py with all paths correctly specified. Then you will need to run model_train.py, and finally you will need to alter main.py to run with the new model.

## Examples
<img src="Examples/raw/post.jpg" width="200"> <img src="Examples/contrast/post.jpg_word_0.png" width="200"> Transcribed as pos
<br>
<img src="Examples/raw/because.jpg" width="200"> <img src="Examples/contrast/because.jpg_word_0.png" width="200"> Transcribed as because
<br>
<img src="Examples/raw/Jeremy.jpg" width="200"> <img src="Examples/contrast/Jeremy.jpg_word_0.png" width="200"> Transcribed as seremg
<br>
<img src="Examples/raw/jumped.jpg" width="200"> <img src="Examples/contrast/jumped.jpg_word_0.png" width="200"> Transcribed as jumped
<br>
<img src="Examples/raw/test.jpg" width="200"> <img src="Examples/contrast/test.jpg_word_0.png" width="200"> Transcribed as test
<br>
<img src="Examples/raw/that.jpg" width="200"> <img src="Examples/contrast/that.jpg_word_0.png" width="200"> Transcribed as thah
<br>
<img src="Examples/raw/This.jpg" width="200"> <img src="Examples/contrast/This.jpg_word_0.png" width="200"> Transcribed as This
<br>
<img src="Examples/raw/wanted.jpg" width="200"> <img src="Examples/contrast/wanted.jpg_word_0.png" width="200"> Transcribed as wranted
<br>
<img src="Examples/raw/sphinx.jpg" width="200"> <img src="Examples/contrast/sphinx.jpg_word_0.png" width="200"> Transcribed as sphnx
<br>
<img src="Examples/raw/science.jpg" width="200"> <img src="Examples/contrast/science.jpg_word_0.png" width="200"> Transcribed as science
<br>
<img src="Examples/raw/1969.jpg" width="200"> <img src="Examples/contrast/1969.jpg_word_0.png" width="200"> Transcribed as 1961
<br>
<img src="Examples/raw/house.jpg" width="200"> <img src="Examples/contrast/house.jpg_word_0.png" width="200"> Transcribed as mouse
<br>

## License

[MIT](https://choosealicense.com/licenses/mit/)
