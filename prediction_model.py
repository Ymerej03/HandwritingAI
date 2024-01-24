import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

import matplotlib.pyplot as plt
import os
# from https://github.com/pythonlessons/mltu/tree/main/Tutorials/03_handwriting_recognition and
# https://www.youtube.com/watch?v=WhRC31SlXzA same code ones more of a guide

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

if __name__ == "__main__":
    # import pandas as pd
    # from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("Models/03_handwriting_recognition/202401241905/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    items = os.listdir("individual_words")
    for item in items:
        item_path = os.path.join("individual_words", item)
        image = cv2.imread(item_path)
        prediction_text = model.predict(image)
        plt.title(prediction_text)
        plt.imshow(image, cmap='gray')
        plt.show()

    # df = pd.read_csv("Models/03_handwriting_recognition/202301111911/val.csv").values.tolist()
    #
    # accum_cer = []
    # for image_path, label in tqdm(df):
    #     image = cv2.imread(image_path)
    #
    #     prediction_text = model.predict(image)
    #
    #     cer = get_cer(prediction_text, label)
    #     print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")
    #
    #     accum_cer.append(cer)
    #
    #     # resize by 4x
    #     image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
    #     cv2.imshow("Image", image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    # print(f"Average CER: {np.average(accum_cer)}")