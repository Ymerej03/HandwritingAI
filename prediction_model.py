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
    # Model I trained with fewer epochs
    configs1 = BaseModelConfigs.load("Models/03_handwriting_recognition/202401241905/configs.yaml")
    model1 = ImageToWordModel(model_path=configs1.model_path, char_list=configs1.vocab)
    words1 = image_processing.extract_words_from_image('sample_handwriting/common_a.jpeg', True)
    transcription1 = ""
    for i in range(len(words1)):
        if np.all(words1[i] == -999):
            transcription1 += "\n"
        else:
            image = cv2.cvtColor(words1[i], cv2.COLOR_GRAY2BGR)
            prediction_text = model1.predict(image)
            transcription1 = transcription1 + prediction_text + " "
    print("my transcription \n")
    print(transcription1)

    # Pre-Trained model
    configs2 = BaseModelConfigs.load("Models/03_handwriting_recognition/202301111911/configs.yaml")
    model2 = ImageToWordModel(model_path=configs2.model_path, char_list=configs2.vocab)
    words2 = image_processing.extract_words_from_image('sample_handwriting/common_a.jpeg', True)
    transcription2 = ""
    for i in range(len(words2)):
        if np.all(words2[i] == -999):
            transcription2 += "\n"
        else:
            image = cv2.cvtColor(words2[i], cv2.COLOR_GRAY2BGR)
            prediction_text = model2.predict(image)
            transcription2 = transcription2 + prediction_text + " "
    print("\n\ntheir transcription\n")
    print(transcription2)


if __name__ == "__main__":
    main()
