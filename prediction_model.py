import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

import matplotlib.pyplot as plt
import os
# from https://github.com/pythonlessons/mltu/tree/main/Tutorials/03_handwriting_recognition and
# https://www.youtube.com/watch?v=WhRC31SlXzA same code ones more of a guide
import image_processing


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text


def main():
    from mltu.configs import BaseModelConfigs
    configs = BaseModelConfigs.load("Models/03_handwriting_recognition/202401241905/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    words = image_processing.extract_words_from_image('sample_handwriting/image1.jpeg', True)
    transcription = ""
    for i in range(len(words)):
        if np.all(words[i] == -999):
            transcription += "\n"
        else:
            image = cv2.cvtColor(words[i], cv2.COLOR_GRAY2BGR)
            prediction_text = model.predict(image)
            transcription = transcription + prediction_text + " "
    print(transcription)


if __name__ == "__main__":
    main()
