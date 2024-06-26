import cv2
import typing
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.metadata["vocab"])[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    model = ImageToWordModel(model_path="Models\\08_handwriting_recognition_torch\\202406191516\model.onnx")
    data_dir="C:\Workplace\Python\\rtrp\demo"
    accum_cer = []
    for dir_path,dir_names,file_names in tqdm(os.walk(data_dir)):
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)
            image = cv2.imread(file_path)
            prediction_text = model.predict(image)
            label=file_name.split("_")[0]
            cer = get_cer(prediction_text, label)
            print(f"Image: {file_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

            accum_cer.append(cer)

    print(f"Average CER: {np.average(accum_cer)}")