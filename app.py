from flask import Flask, render_template, request
import re
import string
import numpy as np
import shutil
from predict import ImageToWordModel
from recommend import recommend
import cv2
app = Flask(__name__)
@app.route('/')
def home():
   return render_template('home.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      img_to_test = request.files['image']
      img = cv2.imdecode(np.frombuffer(img_to_test.read(), np.uint8), 1)
      #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img_to_test = "static/assets/img/user.jpeg"
      model = ImageToWordModel(model_path="Models\\08_handwriting_recognition_torch\\202406191516\model.onnx")
      cv2.imwrite(img_to_test,img)
      image = cv2.imread(img_to_test)
      prediction_text = model.predict(image)
      alternate=recommend(prediction_text)
      print(prediction_text,alternate)
   return render_template("result.html",file_name=img_to_test,prediction=prediction_text,alternate=alternate)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000,debug=True)