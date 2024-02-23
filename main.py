from mltu.configs import BaseModelConfigs

import image_processing
import cv2
import numpy as np
import os
import prediction_model


def main():
    image_path = input("Input the path to the image you want to transcribe:\n")
    while not os.path.exists(image_path):
        image_path = input("Please provide a valid path to the image you want to transcribe:\n")

    model_chosen = False
    while not model_chosen:
        model_to_use = input("Do you want to use the specific or general model?\n")
        if model_to_use == "specific":
            configs = BaseModelConfigs.load(
                "Trained_Models/03_handwriting_recognition/my_handwriting_large_WER/configs.yaml")
            model_chosen = True
        elif model_to_use == "general":
            configs = BaseModelConfigs.load(
                "Trained_Models/03_handwriting_recognition/pre_trained_IAM/configs.yaml")
            model_chosen = True
        else:
            print("Please enter 'specific' or 'general'")

    model = prediction_model.ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    words = image_processing.extract_words_from_image(image_path, True)
    transcription = ""
    for i in range(len(words)):
        if np.all(words[i] == -999):
            transcription += "\n"
        else:
            image = cv2.cvtColor(words[i], cv2.COLOR_GRAY2BGR)
            prediction_text = model.predict(image)
            transcription = transcription + prediction_text + " "
    print(f"\n\nTranscription using {model_to_use} model\n")
    print(transcription)


if __name__ == "__main__":
    main()
