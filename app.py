import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import base64

def add_background_image(image_file):
    with open(image_file, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode()

    css_style = f"""
     <style>
     .stApp {{
         background-image: url("data:image/jpg;base64,{base64_image}");
         background-size: cover;
         background-position: center;
         background-repeat: no-repeat;
     }}
     </style>
     """
    st.markdown(css_style, unsafe_allow_html=True)

working_dir=os.path.dirname(os.path.abspath(__file__))
model_path=f"{working_dir}/nail_model.h5"

model=tf.keras.models.load_model(model_path)

class_indices=json.load(open(f"{working_dir}/class_indices.json"))

def load_and_preprocess_image(image_path,target_size=(224,224)):
    img=Image.open(image_path)
    img=img.resize(target_size)
    img_array=np.array(img)
    img_array=np.expand_dims(img_array,axis=0)
    img_array=img_array.astype('float32')/255
    return img_array

def predict_image_class(model,image_path,class_indices):
    preprocessed_img=load_and_preprocess_image(image_path)
    predictions=model.predict(preprocessed_img)
    predicted_class_index=np.argmax(predictions,axis=1)[0]
    predicted_class_name=class_indices[str(predicted_class_index)]
    return predicted_class_name

add_background_image("image.jpg")
st.title(":red[🩺💅🏻NAIL DISEASE DETECTION]")

uploaded_image=st.file_uploader("upload an image...",type=["jpg","jpeg","png"])

if uploaded_image is not None:
    image=Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img=image.resize((150,150))
        st.image(resized_img)

    with col2:
        if st.button("classify"):
            prediction=predict_image_class(model,uploaded_image,class_indices)
            st.success(f'prediction: {str(prediction)}')