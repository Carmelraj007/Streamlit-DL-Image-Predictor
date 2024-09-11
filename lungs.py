import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
model=load_model("C:/Users/carme/OneDrive/ドキュメント/Deep Learning/chest_disease.h5")
model.compile(optimizer="Adam", loss="binary_crossentropy",metrics=["accuracy"])
st.title("Lung disease prediction")
load_File = st.file_uploader("Upload")
btn = st.button("Submit")
if btn ==True:
    img=image.load_img(load_File,target_size=(150,150,1),color_mode='grayscale')
    image_arr=image.img_to_array(img)
    img_array=np.expand_dims(image_arr,axis=0)
    pic_file=img_array/255.0
    predictions=model.predict(pic_file)
    if predictions<0.5:
        st.write("negative")
    else:
        st.write("positive")