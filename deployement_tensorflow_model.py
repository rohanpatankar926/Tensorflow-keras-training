import base64
from pyexpat import model
import numpy as np
import io
from PIL import Image
from yaml import load
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify
from flask import Flask

app=Flask(__name__)

class Deployment:
    def __init__(self):
        self.model_path="webapp/model/model.h5"
    
    def get_model(self):
        global model
        self.model=load_model(self.model_path)
        print("Model loaded successfully")
        
    def preprocess_image(self,image,target_size):
        if self.image.mode!="RGB":
            self.image=self.image.convert("RGB")
        self.image=image.resize(target_size)
        self.image=img_to_array(image)
        self.mage=np.expand_dims(image,axis=0)
        return self.image
    
    print("Loading keras model.../")
    
@app.route("/predict",methods=["POST"])
def predict():
    message=request.get_json()
    encoded_img=message["image"]
    decoded_img=base64.b64decode(encoded_img)
    image=Image.open(io.BytesIO(decoded_img))
    prediction=model.predict(image).tolist()
    response={
        "prediction":{
            "dog":prediction[0][0],
            "cat":prediction[0][1]          
    } }
    return jsonify(response)
